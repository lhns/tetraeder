use tetraeder_ggml_sys as sys;

#[test]
fn cpu_add_two_tensors() {
    // SAFETY: all pointers come from ggml and are used only while their owning context/backend
    // is alive; the context is created with no_alloc so tensor data lives in the backend buffer.
    unsafe {
        let backend = sys::ggml_backend_cpu_init();
        assert!(!backend.is_null(), "cpu backend init failed");

        let params = sys::ggml_init_params {
            mem_size: sys::ggml_tensor_overhead() * 8 + sys::ggml_graph_overhead() + 1024,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = sys::ggml_init(params);
        assert!(!ctx.is_null());

        let a = sys::ggml_new_tensor_1d(ctx, sys::GGML_TYPE_F32, 4);
        let b = sys::ggml_new_tensor_1d(ctx, sys::GGML_TYPE_F32, 4);
        let c = sys::ggml_add(ctx, a, b);
        let graph = sys::ggml_new_graph(ctx);
        sys::ggml_build_forward_expand(graph, c);

        let buf = sys::ggml_backend_alloc_ctx_tensors(ctx, backend);
        assert!(!buf.is_null(), "tensor allocation failed");

        let xs = [1.0f32, 2.0, 3.0, 4.0];
        let ys = [10.0f32, 20.0, 30.0, 40.0];
        let nbytes = std::mem::size_of_val(&xs);
        sys::ggml_backend_tensor_set(a, xs.as_ptr().cast(), 0, nbytes);
        sys::ggml_backend_tensor_set(b, ys.as_ptr().cast(), 0, nbytes);

        let status = sys::ggml_backend_graph_compute(backend, graph);
        assert_eq!(status, sys::GGML_STATUS_SUCCESS);

        let mut out = [0.0f32; 4];
        sys::ggml_backend_tensor_get(c, out.as_mut_ptr().cast(), 0, nbytes);
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);

        sys::ggml_backend_buffer_free(buf);
        sys::ggml_free(ctx);
        sys::ggml_backend_free(backend);
    }
}

#[test]
fn llama_backend_init_free() {
    // SAFETY: init/free are paired and take no pointers.
    unsafe {
        sys::llama_backend_init();
        sys::llama_backend_free();
    }
}
