extern crate cc;

fn main() {
    println!("cargo:rerun-if-changed=clib/sparse/spmat.c");
    cc::Build::new()
        .file("clib/sparse/spmat.c")
        .compile("mycfuncs");

    //println!("cargo:rustc-link-search=native=/usr/local/lib");
    //println!("cargo:rustc-link-lib=static=blis");
}