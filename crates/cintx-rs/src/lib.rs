//! Facade crate for the cintx workspace.

pub mod api;
pub mod builder;
pub mod prelude;

pub use api::{FacadeError, SessionQuery, SessionRequest, TypedEvaluationOutput};
pub use builder::SessionBuilder;

#[cfg(feature = "unstable-source-api")]
pub use api::unstable;
