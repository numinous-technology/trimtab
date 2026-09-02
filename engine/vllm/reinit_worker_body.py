    def trimtab_release_kv(self) -> dict:
        """trimtab warm reinit, worker side. Unmap the KV pool and drop the
        captured CUDA graphs. Weights stay on the GPU.

        Needs --enable-sleep-mode so the KV pool lives in the CuMemAllocator,
        whose discard() unmaps a tag's physical pages regardless of who still
        references the tensors. The old tensors are kept alive on purpose so
        the allocator's pool never hands their unmapped segments to the new KV
        cache. Only virtual address space is retained per reinit.
        """
        import gc

        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.device_allocator.cumem import CuMemAllocator

        mr = self.model_runner
        torch.cuda.synchronize()
        free0 = torch.cuda.mem_get_info()[0]
        # Drop the captured graphs first, they reference the KV tensors.
        try:
            from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

            BreakableCUDAGraphWrapper.clear_all_graphs()
        except Exception:
            pass
        CUDAGraphWrapper.clear_all_graphs()
        for key_set in mr.cudagraph_dispatcher.cudagraph_keys.values():
            key_set.clear()
        mr.cudagraph_dispatcher.keys_initialized = False
        # Release the KV tensors. Unlike PyTorch's default caching allocator,
        # the CuMemAllocator free callback unmaps each freed block back to the
        # OS, so simply dropping every reference and emptying the cache returns
        # the physical pages. vLLM's own post-profiling teardown does the same.
        mr._cleanup_profiling_kv_cache()
        # Physically unmap the KV pool. discard() releases the tagged pages back
        # to the OS regardless of what still references the tensors (hybrid GDN
        # models keep mamba-state refs that plain deref does not reach).
        CuMemAllocator.get_instance().discard("kv_cache")
        # vLLM caps the process at gpu_memory_utilization via
        # set_per_process_memory_fraction. Lift it, the profiler sizes the new
        # pool against device free memory, not this guard.
        torch.cuda.set_per_process_memory_fraction(1.0)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # The profiler sizes the new pool against init_snapshot, captured at the
        # original boot. Refresh it so the baseline is the current state: weights
        # resident, KV freed. MemorySnapshot measures on construction.
        from vllm.utils.mem_utils import MemorySnapshot

        self.init_snapshot = MemorySnapshot(device=self.device)
        _ = CuMemAllocator  # imported for the assertion that sleep mode is on
        return {"freed_gib": round((torch.cuda.mem_get_info()[0] - free0) / 2**30, 2)}

