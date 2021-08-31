#![allow(clippy::upper_case_acronyms)]

#[cfg(feature = "gpu")]
use rust_gpu_tools::opencl;

#[derive(thiserror::Error, Debug)]
pub enum GPUError {
    #[error("GPUError: {0}")]
    Simple(&'static str),
    #[cfg(feature = "gpu")]
    #[error("OpenCL Error: {0}")]
    OpenCL(#[from] opencl::GPUError),
    #[cfg(feature = "gpu")]
    #[error("GPU taken by a high priority process!")]
    GPUTaken,
    #[cfg(feature = "gpu")]
    #[error("No kernel is initialized!")]
    KernelUninitialized,
    #[error("GPU accelerator is disabled!")]
    GPUDisabled,
    #[error("SynthesisError: {0}")]
    Synthesis(#[from] Box<crate::SynthesisError>),
}

pub type GPUResult<T> = std::result::Result<T, GPUError>;

#[cfg(feature = "gpu")]
impl From<Box<dyn std::any::Any + std::marker::Send>> for GPUError {
    fn from(e: Box<dyn std::any::Any + std::marker::Send>) -> Self {
        match e.downcast::<Self>() {
            Ok(err) => *err,
            Err(_) => GPUError::Simple("An unknown GPU error happened!"),
        }
    }
}

impl From<crate::SynthesisError> for GPUError {
    fn from(e: crate::SynthesisError) -> Self {
        GPUError::Synthesis(Box::new(e))
    }
}
