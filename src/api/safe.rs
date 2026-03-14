use crate::contracts::{BasisSet, Operator, Representation};
use crate::diagnostics::QueryResult;
use crate::runtime::{WorkspaceQuery, WorkspaceQueryOptions, query_workspace_safe};

pub fn query_workspace(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    options: &WorkspaceQueryOptions,
) -> QueryResult<WorkspaceQuery> {
    query_workspace_safe(basis, operator, representation, shell_tuple, options)
}
