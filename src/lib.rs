use std::ffi::c_void;
use std::os::raw::{c_int, c_longlong};

extern "C" {
    fn kmeans_centroids_alloc(centroids: *const f32, k: c_longlong, dim: c_longlong)
        -> *mut c_void;
    fn kmeans_centroids_free(handle: *mut c_void);
    fn kmeans_centroids_ptr(handle: *mut c_void) -> *mut c_void;
    fn kmeans_assign(
        handle: *mut c_void,
        data: *const f32,
        n: c_longlong,
        dim: c_longlong,
        metric: c_int,
        batch: c_longlong,
        out_idx: *mut c_longlong,
    );
}

pub struct Centroids {
    handle: *mut c_void,
}

impl Centroids {
    pub fn new(centroids: &[f32], k: i64, dim: i64) -> Self {
        unsafe {
            let handle = kmeans_centroids_alloc(centroids.as_ptr(), k as _, dim as _);
            Self { handle }
        }
    }

    pub fn device_ptr(&self) -> *mut c_void {
        unsafe { kmeans_centroids_ptr(self.handle) }
    }

    pub fn assign(&self, data: &[f32], n: i64, dim: i64, metric: i32, batch: i64, out: &mut [i64]) {
        assert_eq!(data.len(), (n * dim) as usize);
        assert_eq!(out.len(), n as usize);
        unsafe {
            kmeans_assign(
                self.handle,
                data.as_ptr(),
                n as _,
                dim as _,
                metric as _,
                batch as _,
                out.as_mut_ptr(),
            );
        }
    }
}

impl Drop for Centroids {
    fn drop(&mut self) {
        unsafe { kmeans_centroids_free(self.handle) }
    }
}
