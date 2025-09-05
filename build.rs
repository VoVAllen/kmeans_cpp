fn main() {
    let torch_include = std::env::var("TORCH_INCLUDE").expect("TORCH_INCLUDE env var not set");
    let torch_lib = std::env::var("TORCH_LIB").expect("TORCH_LIB env var not set");

    cc::Build::new()
        .cpp(true)
        .file("cpp/assignment.cpp")
        .include(&torch_include)
        .include(format!("{}/torch/csrc/api/include", torch_include))
        .flag("-std=c++17")
        .compile("assignment");

    println!("cargo:rustc-link-search=native={}", torch_lib);
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");
    println!("cargo:rustc-link-lib=dylib=torch_cuda");
}
