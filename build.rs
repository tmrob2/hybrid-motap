extern crate cc;
use std::env;

fn get_var(name: &str) -> String {
    match env::var(name) {
        Ok(v) => { v }
        Err(e) => { panic!("${} is not set({})", name, e)}
    }
}

fn main() {
    println!("cargo:rerun-if-changed=clib/cuda/motap.cu");
    cc::Build::new()
        .file("clib/sparse/spmat.c")
        .compile("mycfuncs");

    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag(&*format!("arch={},code={}", get_var("CUDA_ARCH"), get_var("CUDA_CM")))
        .file("clib/cuda/motap.cu")
        //.file("myclib/test_cublas.cu")
        .compile("libcudatest.a");

    println!("cargo:rustc-link-search=native={}", get_var("CUDA_LIB"));
    //println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cusparse");
    println!("cargo:rustc-link-lib=dylib=glpk");

    //println!("cargo:rustc-link-search=native=/usr/local/lib");
    //println!("cargo:rustc-link-lib=static=blis");
}