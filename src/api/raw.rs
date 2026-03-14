use crate::contracts::{BasisSet, Operator, Representation};
use crate::diagnostics::QueryResult;
use crate::runtime::{WorkspaceQuery, WorkspaceQueryOptions, query_workspace_raw};

pub fn query_workspace(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    dims_override: Option<&[usize]>,
    options: &WorkspaceQueryOptions,
) -> QueryResult<WorkspaceQuery> {
    query_workspace_raw(
        basis,
        operator,
        representation,
        shell_tuple,
        dims_override,
        options,
    )
}
