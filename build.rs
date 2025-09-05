fn main() {
    let cutlass = std::env::var("CUTLASS_PATH").expect("CUTLASS_PATH not set");
    cc::Build::new()
        .cuda(true)
        .file("cpp/assignment.cu")
        .flag("-std=c++17")
        .include(format!("{}/include", cutlass))
        .compile("assignment");

    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
