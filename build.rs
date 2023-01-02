extern crate cc;

fn main() {
    println!("cargo:rerun-if-changed=clib/cuda/motap.cu");
    cc::Build::new()
        .file("clib/sparse/spmat.c")
        .compile("mycfuncs");

    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_86,code=sm_86")
        .file("clib/cuda/motap.cu")
        //.file("myclib/test_cublas.cu")
        .compile("libcudatest.a");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/");
    //println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cusparse");
    println!("cargo:rustc-link-lib=dylib=glpk");

    //println!("cargo:rustc-link-search=native=/usr/local/lib");
    //println!("cargo:rustc-link-lib=static=blis");
}