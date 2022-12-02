import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
import torch.fx.traceback as fx_traceback
from torch import fx
from torch.fx.node import Node
# from torch.distributed.pipeline.sync import Pipe

from torch._dynamo.optimizations import BACKENDS
from .distributed import Bucket
from torch._inductor.compile_fx import compile_fx

log = logging.getLogger(__name__)

PIPE_BACKEND = ['inductor']

class PipelineOptimizer:
    """
    Very basic & hand-written 2-GPU auto-pipeliner
    """
    def __init__(
        self,
        backend_compile_fn,
        first_bucket_cap: Optional[int] = None,
    ):
        self.backend_compile_fn = backend_compile_fn

    def _ignore_parameter(self, parameter):
        return hasattr(parameter, "_ddp_ignored") and parameter._ddp_ignored

    def compile_fn(self, gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
        gpu_n = 2

        total_bytes = 0
        for node in gm.graph.nodes:
            if node.op == "call_module":
                target = gm.get_submodule(node.target)
                for name, p in target.named_parameters():
                    param = target.get_parameter(name)
                    if p.requires_grad and not self._ignore_parameter(param):
                        total_bytes += p._storage().nbytes()
            elif node.op == "get_attr":
                maybe_param = getattr(gm, node.target)
                if maybe_param.requires_grad and not self._ignore_parameter(
                    maybe_param
                ):
                    total_bytes += maybe_param._storage().nbytes()

        bucket_bytes = total_bytes // gpu_n

        # 1: compute the partition map according to bucket logic
        buckets = [Bucket()]  # (size, param_names)
        for node in gm.graph.nodes:
            if node.op in ("output", "placeholder"):
                continue
            
            buck = buckets[len(buckets) - 1]

            if buck.size >= bucket_bytes and len(buckets) < gpu_n:
                buckets.append(Bucket())

            if node.op == "call_module":
                target = gm.get_submodule(node.target)
                for name, p in target.named_parameters():
                    param = target.get_parameter(name)
                    if p.requires_grad and not self._ignore_parameter(param):
                        buck.size += p._storage().nbytes()
                        buck.params.append(f"{node.target}_{name}")
                        buck.param_ids.append(id(param))
            elif node.op == "get_attr":
                maybe_param = getattr(gm, node.target)
                if maybe_param.requires_grad and not self._ignore_parameter(
                    maybe_param
                ):
                    buck.size += maybe_param._storage().nbytes()
                    buck.params.append(node.target)
                    buck.param_ids.append(id(maybe_param))

            # All nodes have to be mapped to a bucket, even if they don't have their own params
            # Ignored params still end up in buckets, we just don't count them towards the capacity
            buck.nodes.append(node)

        # stash buckets for testing/debugging purposes
        self.buckets = buckets

        if len(buckets) == 1:
            # bypass split/fuse logic if there is only one bucket
            return self.backend_compile_fn(gm, example_inputs)

        # 2: partition the graphmodule according to bucket capacity
        partition_map = {}
        for idx, b in enumerate(buckets):
            for node in b.nodes:
                partition_map[node] = idx

        split_gm = fx.passes.split_module.split_module(
            gm, None, lambda node: partition_map[node]
        )

        debug_str = (
            f"\n---orig graph---\n{gm.graph}\n"
            + f"\n---split graph---\n{split_gm.graph}\n"
        )
        for name, module in split_gm.named_modules():
            if "." not in name and len(name):
                # only print the submod graphs, not their children
                debug_str += f"\n---{name} graph---\n{module.graph}\n"
        debug_str += "\n---------------\n"
        print(debug_str)

        # pipelined modules
        submodules = []

        # 3: compile each of the partitioned submodules using the user-provided compiler
        class SubmodCompiler(torch.fx.interpreter.Interpreter):
            def __init__(self, module, compiler):
                super().__init__(module)
                self.compiler = compiler
                self.submodule_idx = 0

            def compile_submod(self, submod, args, kwargs):
                """
                Compile the submodule,
                using a wrapper to make sure its output is always a tuple,
                which is required by AotAutograd based compilers
                """
                assert len(kwargs) == 0, "We assume only args for these modules"

                class WrapperModule(torch.nn.Module):
                    def __init__(self, compiled_submod, unwrap_singleton_tuple):
                        super().__init__()
                        self.compiled_submod = compiled_submod
                        self.unwrap_singleton_tuple = unwrap_singleton_tuple

                    def forward(self, *args):
                        x = self.compiled_submod(*args)
                        # TODO(whc)
                        # for some reason the isinstance check is necessary if I split one node per submod
                        # - even though I supposedly wrapped the output in a tuple in those cases, the real
                        # compiled module was still returning a tensor
                        if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
                            return x[0]
                        return x

                unwrap_singleton_tuple = False
                for sn in submod.graph.nodes:
                    if sn.op == "output":
                        if not isinstance(sn.args[0], tuple):
                            unwrap_singleton_tuple = True
                            sn.args = (sn.args,)
                submod.recompile()

                wrapper = WrapperModule(
                    self.compiler(submod, args),
                    unwrap_singleton_tuple,
                )
                return wrapper

            def run_node(self, n: Node) -> Any:
                with fx_traceback.append_stack_trace(n.stack_trace):
                    # TODO: 

                    args, kwargs = self.fetch_args_kwargs_from_env(n)
                    assert isinstance(args, tuple)
                    assert isinstance(kwargs, dict)

                    # modify the currently running FX graph
                    # maybe this isn't sound in general, but only changing the target of a node might be ok?
                    if n.op == "call_module":
                        submod = self.fetch_attr(n.target)
                        idx = self.submodule_idx
                        device = f'cuda:{idx}'
                        # log.debug(f"\n---{n.target} graph---\n" + str(submod.graph))
                        compiled_submod = self.compile_submod(submod, args, kwargs)
                        
                        submodules.append(compiled_submod)

                        self.module.delete_submodule(n.target)
                        n.target = "compiled_" + n.target
                        self.module.add_submodule(n.target, compiled_submod)

                        # TODO: prepend nodes of .to('cuda:n') for every args & kwargs
                        new_args = []
                        for arg in n.args:
                            new_arg = arg
                            if isinstance(arg, Node):
                                # TODO: check arg is raw tensor
                                new_arg = Node(arg.graph, f'{arg.name}_moved', 'call_method', 'to', (arg, device), dict())
                                n.prepend(new_arg)
                            new_args.append(new_arg)
                        n.args = tuple(new_args)

                        self.submodule_idx += 1
                        
                    # if n.op == 'call_function' and n.name not in self.names:
                    #     # insert no-op identity function
                    #     origin_name = n.name
                    #     self.names.add(origin_name)
                    #     n.name = f'temp_{n.name}'
                    #     n.append(Node(n.graph, origin_name, 'call_function', lambda x: x, (n, ), dict()))
                    #     print(n.target, type(n.target))
                    #     print(n.args[0], type(n.args[0]), n.args[0] == self.last_submodule)
                    # then we execute the modified node using the usual logic
                    return getattr(self, n.op)(n.target, args, kwargs)

        submod_compiler = SubmodCompiler(split_gm, self.backend_compile_fn)
        submod_compiler.run(*example_inputs)
        split_gm.recompile()

        for i, submod in enumerate(submodules):
            submod.to(f'cuda:{i}')

        # log.debug("\n---final graph---\n" + str(split_gm.graph) + "\n---------------\n")

        # cuda_modules = []
        # for i, mod in enumerate(submodules):
        #     cuda_modules.append(mod.to(f"cuda:{i}"))

        # log.info(f'modules cnt: {len(cuda_modules)}')

        # class ReturnToZero(torch.fx.GraphModule):
        #     def __init__(self, gm, mod1, mod2):
        #         super().__init__(gm, gm.graph)
        #         self.mod1 = mod1
        #         self.mod2 = mod2
            
        #     def forward(self, *args, **kwargs):
        #         # x = self.submodule(*args, **kwargs)
        #         # return x.local_value().to(f"cuda:0")
        #         print("args z: ", args)
        #         print("kwargs z:", kwargs)
        #         x = self.mod1(*args, **kwargs)
        #         x = x.to("cuda:1")
        #         x = self.mod2(x)
        #         return (x.to("cuda:0"), )

        # # TODO: set chunks from config        
        # # pipe = Pipe(torch.nn.Sequential(*cuda_modules), chunks = 4)
        # # zero = ReturnToZero(pipe)
        # print("make zero")
        # zero = ReturnToZero(split_gm, *cuda_modules)
        # return zero.forward

        print("\n---final graph---\n" + str(split_gm.graph) + "\n---------------\n")

        return split_gm

def pipeline_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    backend = PIPE_BACKEND[0]
    if backend == 'inductor':
        backend = compile_fx
    else:
        backend = BACKENDS[backend]
    opt = PipelineOptimizer(backend)
    return opt.compile_fn(gm, example_inputs)
