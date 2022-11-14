extern crate cc;

fn main() {
    cc::Build::new()
        .file("clib/intelmkl/multiobj.c")
        .file("clib/intelmkl/tests.c")
        //.file("myclib/test_array_csr.c")
        .include("/opt/intel/oneapi/mkl/2022.2.1/include/")
        .compile("mycfuncs");

    /*println!("cargo:rustc-link-search=native=/opt/intel/oneapi/mkl/2022.2.1/lib/intel64");
    println!("cargo:rustc-link-lib=mkl_intel_ilp64");
    println!("cargo:rustc-link-lib=mkl_sequential");
    println!("cargo:rustc-link-lib=mkl_core");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=pthread")
    */
    println!("cargo:rustc-link-lib=mkl_rt");
}