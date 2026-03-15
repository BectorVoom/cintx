# libcint Rust モジュール詳細設計マニフェスト
## 1. 生成サマリ
- 総候補数: **86**
- In scope: **84**
- Out of scope: **2**
- 生成先ディレクトリ: `/mnt/data/libcint_module_designs`
- stub生成数: **0**

## 2. ターゲットモジュール一覧表
| No | Module | Source Path | Category | Scope | Output | 備考 |
|---:|---|---|---|---|---|---|
| 1 | `Cargo` | `Cargo.toml` | Internal support module | In | `/mnt/data/libcint_module_designs/cargo_toml_detailed_design.md` | ビルド境界・feature行列・依存境界・再現性を規定する主要設定ファイルであり、実装成立に直接影響する。 |
| 2 | `rust_toolchain` | `rust-toolchain.toml` | Internal support module | In | `/mnt/data/libcint_module_designs/rust_toolchain_toml_detailed_design.md` | ビルド境界・feature行列・依存境界・再現性を規定する主要設定ファイルであり、実装成立に直接影響する。 |
| 3 | `README` | `README.md` | Internal support module | In | `/mnt/data/libcint_module_designs/readme_md_detailed_design.md` | 互換性表・Mermaid図・利用者向け説明はリリースゲートとAPI可視化に関わるため含める。 |
| 4 | `LICENSE` | `LICENSE` | Candidate out-of-scope | Out | `/mnt/data/libcint_module_designs/license_detailed_design.md` | ソースツリー上の主要ファイルではあるが、前者は法務アセット、後者は本タスクの入力となる上位設計書そのものであり、再実装対象の実装モジュールではない。 |
| 5 | `docs::design::libcint_rust_detailed_design_reviewed` | `docs/design/libcint_rust_detailed_design_reviewed.md` | Candidate out-of-scope | Out | `/mnt/data/libcint_module_designs/docs_design_libcint_rust_detailed_design_reviewed_md_detailed_design.md` | ソースツリー上の主要ファイルではあるが、前者は法務アセット、後者は本タスクの入力となる上位設計書そのものであり、再実装対象の実装モジュールではない。 |
| 6 | `docs::design::api_manifest` | `docs/design/api_manifest.csv` | Generated artifact | In | `/mnt/data/libcint_module_designs/docs_design_api_manifest_csv_detailed_design.md` | 生成物ではあるが、API棚卸し・互換性報告の正本として上位設計で明示されているため含める。 |
| 7 | `docs::design::diagrams` | `docs/design/diagrams.md` | Internal support module | In | `/mnt/data/libcint_module_designs/docs_design_diagrams_md_detailed_design.md` | 互換性表・Mermaid図・利用者向け説明はリリースゲートとAPI可視化に関わるため含める。 |
| 8 | `docs::compatibility` | `docs/compatibility.md` | Internal support module | In | `/mnt/data/libcint_module_designs/docs_compatibility_md_detailed_design.md` | 互換性表・Mermaid図・利用者向け説明はリリースゲートとAPI可視化に関わるため含める。 |
| 9 | `libcint_rs::Cargo` | `crates/libcint-rs/Cargo.toml` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_rs_cargo_toml_detailed_design.md` | ビルド境界・feature行列・依存境界・再現性を規定する主要設定ファイルであり、実装成立に直接影響する。 |
| 10 | `libcint_rs::src::lib` | `crates/libcint-rs/src/lib.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_rs_src_lib_rs_detailed_design.md` | 公開境界・再エクスポート・互換レイヤを構成するため。 |
| 11 | `libcint_rs::src::api` | `crates/libcint-rs/src/api.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_rs_src_api_rs_detailed_design.md` | 安全な公開API境界を担うため。 |
| 12 | `libcint_rs::src::builder` | `crates/libcint-rs/src/builder.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_rs_src_builder_rs_detailed_design.md` | 公開境界・再エクスポート・互換レイヤを構成するため。 |
| 13 | `libcint_rs::src::prelude` | `crates/libcint-rs/src/prelude.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_rs_src_prelude_rs_detailed_design.md` | 公開境界・再エクスポート・互換レイヤを構成するため。 |
| 14 | `libcint_core::Cargo` | `crates/libcint-core/Cargo.toml` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_core_cargo_toml_detailed_design.md` | ビルド境界・feature行列・依存境界・再現性を規定する主要設定ファイルであり、実装成立に直接影響する。 |
| 15 | `libcint_core::src::lib` | `crates/libcint-core/src/lib.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_core_src_lib_rs_detailed_design.md` | 中核データ型・表示レイアウト・エラー境界として実装中心に位置する。 |
| 16 | `libcint_core::src::atom` | `crates/libcint-core/src/atom.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_core_src_atom_rs_detailed_design.md` | 型安全な公開ドメインモデルを担う。 |
| 17 | `libcint_core::src::shell` | `crates/libcint-core/src/shell.rs` | Buffer/data-structure module | In | `/mnt/data/libcint_module_designs/crates_libcint_core_src_shell_rs_detailed_design.md` | テンソル表現・shape・workspace・buffer writer・shell tupleを担うデータ構造中核である。 |
| 18 | `libcint_core::src::basis` | `crates/libcint-core/src/basis.rs` | Buffer/data-structure module | In | `/mnt/data/libcint_module_designs/crates_libcint_core_src_basis_rs_detailed_design.md` | テンソル表現・shape・workspace・buffer writer・shell tupleを担うデータ構造中核である。 |
| 19 | `libcint_core::src::env` | `crates/libcint-core/src/env.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_core_src_env_rs_detailed_design.md` | 型安全な公開ドメインモデルを担う。 |
| 20 | `libcint_core::src::operator` | `crates/libcint-core/src/operator.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_core_src_operator_rs_detailed_design.md` | 型安全な公開ドメインモデルを担う。 |
| 21 | `libcint_core::src::tensor` | `crates/libcint-core/src/tensor.rs` | Buffer/data-structure module | In | `/mnt/data/libcint_module_designs/crates_libcint_core_src_tensor_rs_detailed_design.md` | テンソル表現・shape・workspace・buffer writer・shell tupleを担うデータ構造中核である。 |
| 22 | `libcint_core::src::error` | `crates/libcint-core/src/error.rs` | Error-handling module | In | `/mnt/data/libcint_module_designs/crates_libcint_core_src_error_rs_detailed_design.md` | 公開エラー境界またはC ABI error bridgeを定義するため必須。 |
| 23 | `libcint_ops::Cargo` | `crates/libcint-ops/Cargo.toml` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_ops_cargo_toml_detailed_design.md` | ビルド境界・feature行列・依存境界・再現性を規定する主要設定ファイルであり、実装成立に直接影響する。 |
| 24 | `libcint_ops::build` | `crates/libcint-ops/build.rs` | Build script / codegen / FFI support | In | `/mnt/data/libcint_module_designs/crates_libcint_ops_build_rs_detailed_design.md` | manifest生成またはvendored libcint/bindgen連携の成立条件を規定するため、実装スコープに含める。 |
| 25 | `libcint_ops::src::lib` | `crates/libcint-ops/src/lib.rs` | Build script / codegen / FFI support | In | `/mnt/data/libcint_module_designs/crates_libcint_ops_src_lib_rs_detailed_design.md` | 生成manifestとresolverがAPI在庫管理の正本を担う。 |
| 26 | `libcint_ops::src::generated::api_manifest` | `crates/libcint-ops/src/generated/api_manifest.rs` | Generated artifact | In | `/mnt/data/libcint_module_designs/crates_libcint_ops_src_generated_api_manifest_rs_detailed_design.md` | 生成済みmanifestは公開API棚卸しの正本であり、実装上の契約そのものなので含める。 |
| 27 | `libcint_ops::src::generated::api_manifest` | `crates/libcint-ops/src/generated/api_manifest.csv` | Generated artifact | In | `/mnt/data/libcint_module_designs/crates_libcint_ops_src_generated_api_manifest_csv_detailed_design.md` | 生成済みmanifestは公開API棚卸しの正本であり、実装上の契約そのものなので含める。 |
| 28 | `libcint_ops::src::resolver` | `crates/libcint-ops/src/resolver.rs` | Build script / codegen / FFI support | In | `/mnt/data/libcint_module_designs/crates_libcint_ops_src_resolver_rs_detailed_design.md` | 生成manifestとresolverがAPI在庫管理の正本を担う。 |
| 29 | `libcint_runtime::Cargo` | `crates/libcint-runtime/Cargo.toml` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_runtime_cargo_toml_detailed_design.md` | ビルド境界・feature行列・依存境界・再現性を規定する主要設定ファイルであり、実装成立に直接影響する。 |
| 30 | `libcint_runtime::src::lib` | `crates/libcint-runtime/src/lib.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_lib_rs_detailed_design.md` | validator/planner/scheduler/workspace/dispatch/metrics/options は実行制御中核である。 |
| 31 | `libcint_runtime::src::validator` | `crates/libcint-runtime/src/validator.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_validator_rs_detailed_design.md` | validator/planner/scheduler/workspace/dispatch/metrics/options は実行制御中核である。 |
| 32 | `libcint_runtime::src::planner` | `crates/libcint-runtime/src/planner.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_planner_rs_detailed_design.md` | validator/planner/scheduler/workspace/dispatch/metrics/options は実行制御中核である。 |
| 33 | `libcint_runtime::src::scheduler` | `crates/libcint-runtime/src/scheduler.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_scheduler_rs_detailed_design.md` | validator/planner/scheduler/workspace/dispatch/metrics/options は実行制御中核である。 |
| 34 | `libcint_runtime::src::workspace` | `crates/libcint-runtime/src/workspace.rs` | Buffer/data-structure module | In | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_workspace_rs_detailed_design.md` | テンソル表現・shape・workspace・buffer writer・shell tupleを担うデータ構造中核である。 |
| 35 | `libcint_runtime::src::dispatch` | `crates/libcint-runtime/src/dispatch.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_dispatch_rs_detailed_design.md` | validator/planner/scheduler/workspace/dispatch/metrics/options は実行制御中核である。 |
| 36 | `libcint_runtime::src::metrics` | `crates/libcint-runtime/src/metrics.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_metrics_rs_detailed_design.md` | validator/planner/scheduler/workspace/dispatch/metrics/options は実行制御中核である。 |
| 37 | `libcint_runtime::src::options` | `crates/libcint-runtime/src/options.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_options_rs_detailed_design.md` | validator/planner/scheduler/workspace/dispatch/metrics/options は実行制御中核である。 |
| 38 | `libcint_cpu::Cargo` | `crates/libcint-cpu/Cargo.toml` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_cargo_toml_detailed_design.md` | ビルド境界・feature行列・依存境界・再現性を規定する主要設定ファイルであり、実装成立に直接影響する。 |
| 39 | `libcint_cpu::src::lib` | `crates/libcint-cpu/src/lib.rs` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_lib_rs_detailed_design.md` | CPU backendの支援モジュール。 |
| 40 | `libcint_cpu::src::kernels::one_electron` | `crates/libcint-cpu/src/kernels/one_electron.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_kernels_one_electron_rs_detailed_design.md` | CPU計算カーネル本体であり、互換性・性能・メモリ効率に直接影響する。 |
| 41 | `libcint_cpu::src::kernels::two_electron` | `crates/libcint-cpu/src/kernels/two_electron.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_kernels_two_electron_rs_detailed_design.md` | CPU計算カーネル本体であり、互換性・性能・メモリ効率に直接影響する。 |
| 42 | `libcint_cpu::src::kernels::center_2c2e` | `crates/libcint-cpu/src/kernels/center_2c2e.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_kernels_center_2c2e_rs_detailed_design.md` | CPU計算カーネル本体であり、互換性・性能・メモリ効率に直接影響する。 |
| 43 | `libcint_cpu::src::kernels::center_3c1e` | `crates/libcint-cpu/src/kernels/center_3c1e.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_kernels_center_3c1e_rs_detailed_design.md` | CPU計算カーネル本体であり、互換性・性能・メモリ効率に直接影響する。 |
| 44 | `libcint_cpu::src::kernels::center_3c2e` | `crates/libcint-cpu/src/kernels/center_3c2e.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_kernels_center_3c2e_rs_detailed_design.md` | CPU計算カーネル本体であり、互換性・性能・メモリ効率に直接影響する。 |
| 45 | `libcint_cpu::src::kernels::center_4c1e` | `crates/libcint-cpu/src/kernels/center_4c1e.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_kernels_center_4c1e_rs_detailed_design.md` | CPU計算カーネル本体であり、互換性・性能・メモリ効率に直接影響する。 |
| 46 | `libcint_cpu::src::transform::c2s` | `crates/libcint-cpu/src/transform/c2s.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_transform_c2s_rs_detailed_design.md` | cart→sph/spinor変換は互換出力 shape/order に直結する。 |
| 47 | `libcint_cpu::src::transform::c2spinor` | `crates/libcint-cpu/src/transform/c2spinor.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_transform_c2spinor_rs_detailed_design.md` | cart→sph/spinor変換は互換出力 shape/order に直結する。 |
| 48 | `libcint_cpu::src::screening` | `crates/libcint-cpu/src/screening.rs` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_screening_rs_detailed_design.md` | CPU backendの支援モジュール。 |
| 49 | `libcint_cpu::src::simd` | `crates/libcint-cpu/src/simd.rs` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_simd_rs_detailed_design.md` | CPU backendの支援モジュール。 |
| 50 | `libcint_cpu::src::executor` | `crates/libcint-cpu/src/executor.rs` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_executor_rs_detailed_design.md` | CPU backendの支援モジュール。 |
| 51 | `libcint_cubecl::Cargo` | `crates/libcint-cubecl/Cargo.toml` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_cargo_toml_detailed_design.md` | ビルド境界・feature行列・依存境界・再現性を規定する主要設定ファイルであり、実装成立に直接影響する。 |
| 52 | `libcint_cubecl::src::lib` | `crates/libcint-cubecl/src/lib.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_src_lib_rs_detailed_design.md` | GPU backend/転送/常駐cache/特殊化の中核である。 |
| 53 | `libcint_cubecl::src::executor` | `crates/libcint-cubecl/src/executor.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_src_executor_rs_detailed_design.md` | GPU backend/転送/常駐cache/特殊化の中核である。 |
| 54 | `libcint_cubecl::src::kernels` | `crates/libcint-cubecl/src/kernels.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_src_kernels_rs_detailed_design.md` | GPU backend/転送/常駐cache/特殊化の中核である。 |
| 55 | `libcint_cubecl::src::transfer` | `crates/libcint-cubecl/src/transfer.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_src_transfer_rs_detailed_design.md` | GPU backend/転送/常駐cache/特殊化の中核である。 |
| 56 | `libcint_cubecl::src::resident_cache` | `crates/libcint-cubecl/src/resident_cache.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_src_resident_cache_rs_detailed_design.md` | GPU backend/転送/常駐cache/特殊化の中核である。 |
| 57 | `libcint_cubecl::src::specialization` | `crates/libcint-cubecl/src/specialization.rs` | Primary implementation module | In | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_src_specialization_rs_detailed_design.md` | GPU backend/転送/常駐cache/特殊化の中核である。 |
| 58 | `libcint_compat::Cargo` | `crates/libcint-compat/Cargo.toml` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_compat_cargo_toml_detailed_design.md` | ビルド境界・feature行列・依存境界・再現性を規定する主要設定ファイルであり、実装成立に直接影響する。 |
| 59 | `libcint_compat::src::lib` | `crates/libcint-compat/src/lib.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_lib_rs_detailed_design.md` | 公開境界・再エクスポート・互換レイヤを構成するため。 |
| 60 | `libcint_compat::src::raw` | `crates/libcint-compat/src/raw.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_raw_rs_detailed_design.md` | 公開境界・再エクスポート・互換レイヤを構成するため。 |
| 61 | `libcint_compat::src::legacy` | `crates/libcint-compat/src/legacy.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_legacy_rs_detailed_design.md` | 公開境界・再エクスポート・互換レイヤを構成するため。 |
| 62 | `libcint_compat::src::helpers` | `crates/libcint-compat/src/helpers.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_helpers_rs_detailed_design.md` | 公開境界・再エクスポート・互換レイヤを構成するため。 |
| 63 | `libcint_compat::src::optimizer` | `crates/libcint-compat/src/optimizer.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_optimizer_rs_detailed_design.md` | 公開境界・再エクスポート・互換レイヤを構成するため。 |
| 64 | `libcint_compat::src::transform` | `crates/libcint-compat/src/transform.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_transform_rs_detailed_design.md` | 公開境界・再エクスポート・互換レイヤを構成するため。 |
| 65 | `libcint_compat::src::layout` | `crates/libcint-compat/src/layout.rs` | Buffer/data-structure module | In | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_layout_rs_detailed_design.md` | テンソル表現・shape・workspace・buffer writer・shell tupleを担うデータ構造中核である。 |
| 66 | `libcint_capi::Cargo` | `crates/libcint-capi/Cargo.toml` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_capi_cargo_toml_detailed_design.md` | ビルド境界・feature行列・依存境界・再現性を規定する主要設定ファイルであり、実装成立に直接影響する。 |
| 67 | `libcint_capi::src::lib` | `crates/libcint-capi/src/lib.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_capi_src_lib_rs_detailed_design.md` | 公開境界・再エクスポート・互換レイヤを構成するため。 |
| 68 | `libcint_capi::src::errors` | `crates/libcint-capi/src/errors.rs` | Error-handling module | In | `/mnt/data/libcint_module_designs/crates_libcint_capi_src_errors_rs_detailed_design.md` | 公開エラー境界またはC ABI error bridgeを定義するため必須。 |
| 69 | `libcint_capi::src::shim` | `crates/libcint-capi/src/shim.rs` | Public API module | In | `/mnt/data/libcint_module_designs/crates_libcint_capi_src_shim_rs_detailed_design.md` | 公開境界・再エクスポート・互換レイヤを構成するため。 |
| 70 | `libcint_oracle::Cargo` | `crates/libcint-oracle/Cargo.toml` | Internal support module | In | `/mnt/data/libcint_module_designs/crates_libcint_oracle_cargo_toml_detailed_design.md` | ビルド境界・feature行列・依存境界・再現性を規定する主要設定ファイルであり、実装成立に直接影響する。 |
| 71 | `libcint_oracle::build` | `crates/libcint-oracle/build.rs` | Build script / codegen / FFI support | In | `/mnt/data/libcint_module_designs/crates_libcint_oracle_build_rs_detailed_design.md` | manifest生成またはvendored libcint/bindgen連携の成立条件を規定するため、実装スコープに含める。 |
| 72 | `libcint_oracle::src::lib` | `crates/libcint-oracle/src/lib.rs` | Test-support module | In | `/mnt/data/libcint_module_designs/crates_libcint_oracle_src_lib_rs_detailed_design.md` | oracle比較とfixture供給を担う検証専用モジュールであり、比較互換性の成立条件。 |
| 73 | `libcint_oracle::src::compare` | `crates/libcint-oracle/src/compare.rs` | Test-support module | In | `/mnt/data/libcint_module_designs/crates_libcint_oracle_src_compare_rs_detailed_design.md` | oracle比較とfixture供給を担う検証専用モジュールであり、比較互換性の成立条件。 |
| 74 | `libcint_oracle::src::fixtures` | `crates/libcint-oracle/src/fixtures.rs` | Test-support module | In | `/mnt/data/libcint_module_designs/crates_libcint_oracle_src_fixtures_rs_detailed_design.md` | oracle比較とfixture供給を担う検証専用モジュールであり、比較互換性の成立条件。 |
| 75 | `xtask::Cargo` | `xtask/Cargo.toml` | Internal support module | In | `/mnt/data/libcint_module_designs/xtask_cargo_toml_detailed_design.md` | ビルド境界・feature行列・依存境界・再現性を規定する主要設定ファイルであり、実装成立に直接影響する。 |
| 76 | `xtask::src::main` | `xtask/src/main.rs` | Build script / codegen / FFI support | In | `/mnt/data/libcint_module_designs/xtask_src_main_rs_detailed_design.md` | manifest audit・doc生成・oracle更新・bench集約を担う開発運用境界であり、上位設計の検証計画に直結する。 |
| 77 | `xtask::src::manifest_audit` | `xtask/src/manifest_audit.rs` | Build script / codegen / FFI support | In | `/mnt/data/libcint_module_designs/xtask_src_manifest_audit_rs_detailed_design.md` | manifest audit・doc生成・oracle更新・bench集約を担う開発運用境界であり、上位設計の検証計画に直結する。 |
| 78 | `xtask::src::oracle_update` | `xtask/src/oracle_update.rs` | Build script / codegen / FFI support | In | `/mnt/data/libcint_module_designs/xtask_src_oracle_update_rs_detailed_design.md` | manifest audit・doc生成・oracle更新・bench集約を担う開発運用境界であり、上位設計の検証計画に直結する。 |
| 79 | `xtask::src::gen_docs` | `xtask/src/gen_docs.rs` | Build script / codegen / FFI support | In | `/mnt/data/libcint_module_designs/xtask_src_gen_docs_rs_detailed_design.md` | manifest audit・doc生成・oracle更新・bench集約を担う開発運用境界であり、上位設計の検証計画に直結する。 |
| 80 | `xtask::src::bench_report` | `xtask/src/bench_report.rs` | Build script / codegen / FFI support | In | `/mnt/data/libcint_module_designs/xtask_src_bench_report_rs_detailed_design.md` | manifest audit・doc生成・oracle更新・bench集約を担う開発運用境界であり、上位設計の検証計画に直結する。 |
| 81 | `benches::micro_families` | `benches/micro_families.rs` | Benchmark module | In | `/mnt/data/libcint_module_designs/benches_micro_families_rs_detailed_design.md` | 性能・閾値・退行監視は上位設計の必須要件であり、ベンチハーネスは実装対象に含める。 |
| 82 | `benches::macro_molecules` | `benches/macro_molecules.rs` | Benchmark module | In | `/mnt/data/libcint_module_designs/benches_macro_molecules_rs_detailed_design.md` | 性能・閾値・退行監視は上位設計の必須要件であり、ベンチハーネスは実装対象に含める。 |
| 83 | `benches::crossover_cpu_gpu` | `benches/crossover_cpu_gpu.rs` | Benchmark module | In | `/mnt/data/libcint_module_designs/benches_crossover_cpu_gpu_rs_detailed_design.md` | 性能・閾値・退行監視は上位設計の必須要件であり、ベンチハーネスは実装対象に含める。 |
| 84 | `ci::feature_matrix` | `ci/feature-matrix.yml` | Internal support module | In | `/mnt/data/libcint_module_designs/ci_feature_matrix_yml_detailed_design.md` | release gate・feature行列・oracle比較の自動化を担うため、主要支援ファイルとして含める。 |
| 85 | `ci::oracle_compare` | `ci/oracle-compare.yml` | Internal support module | In | `/mnt/data/libcint_module_designs/ci_oracle_compare_yml_detailed_design.md` | release gate・feature行列・oracle比較の自動化を担うため、主要支援ファイルとして含める。 |
| 86 | `ci::gpu_bench` | `ci/gpu-bench.yml` | Internal support module | In | `/mnt/data/libcint_module_designs/ci_gpu_bench_yml_detailed_design.md` | release gate・feature行列・oracle比較の自動化を担うため、主要支援ファイルとして含める。 |

## 3. In-scope / Out-of-scope 分類表
| Scope | Count | 対象例 | 判定方針 |
|---|---:|---|---|
| In scope | 84 | `crates/*/src/*.rs`, `build.rs`, `xtask`, `benches`, `ci`, `docs/compatibility.md` | 上位設計第15章に列挙され、実装・検証・配布・監査の成立に直接関与するものを含める |
| Out of scope | 2 | `LICENSE`, `docs/design/libcint_rust_detailed_design_reviewed.md` | 前者は法務資産、後者は入力上位設計書そのものであり、再実装対象の実装責務を持たない |

## 4. モジュール責務マトリクス（グループ単位）
| Group | 公開API | raw互換 | ドメイン型 | manifest/codegen | runtime計画 | CPU backend | GPU backend | C ABI | oracle/検証 | bench/CI/docs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| workspace | - | - | - | - | - | - | - | - | - | ◯ |
| docs | ◯ | - | - | ◯ | - | - | - | - | ◯ | ◯ |
| facade | ◯ | △ | △ | - | ◯ | - | - | - | - | - |
| core | ◯ | - | ◯ | - | △ | △ | △ | - | - | - |
| ops | △ | △ | - | ◯ | ◯ | - | - | - | ◯ | ◯ |
| runtime | - | △ | △ | - | ◯ | △ | △ | - | △ | △ |
| cpu | - | - | - | - | △ | ◯ | - | - | △ | △ |
| cubecl | - | - | - | - | △ | - | ◯ | - | △ | △ |
| compat | ◯ | ◯ | △ | △ | ◯ | △ | △ | △ | ◯ | - |
| capi | ◯ | ◯ | - | - | - | - | - | ◯ | △ | - |
| oracle | - | △ | - | - | - | - | - | △ | ◯ | △ |
| xtask | - | - | - | ◯ | - | - | - | - | ◯ | ◯ |
| bench | - | - | - | - | △ | △ | △ | - | △ | ◯ |
| ci | - | - | - | △ | - | - | △ | - | ◯ | ◯ |

記号: `◯` 主責務, `△` 補助責務, `-` 非責務

## 5. モジュール依存マトリクス（グループ単位）
| From \ To | workspace | docs | facade | core | ops | runtime | cpu | cubecl | compat | capi | oracle | xtask | bench | ci |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| workspace | - | ◯ | ◯ | ◯ | ◯ | ◯ | ◯ | ◯ | ◯ | ◯ | ◯ | ◯ | ◯ | ◯ |
| docs | △ | - | △ | △ | ◯ | △ | - | - | △ | - | △ | ◯ | △ | ◯ |
| facade | △ | - | - | ◯ | - | ◯ | - | - | ◯ | - | - | - | △ | - |
| core | △ | - | - | - | - | - | - | - | - | - | - | - | - | - |
| ops | △ | - | - | ◯ | - | - | - | - | - | - | - | ◯ | - | - |
| runtime | △ | - | - | ◯ | ◯ | - | △ | △ | △ | - | - | - | △ | - |
| cpu | △ | - | - | ◯ | ◯ | ◯ | - | - | △ | - | - | - | △ | - |
| cubecl | △ | - | - | ◯ | ◯ | ◯ | △ | - | - | - | - | - | △ | - |
| compat | △ | - | - | ◯ | ◯ | ◯ | △ | △ | - | △ | △ | - | - | - |
| capi | △ | - | - | - | - | - | - | - | ◯ | - | - | - | - | - |
| oracle | △ | - | - | - | - | - | - | - | ◯ | △ | - | △ | △ | ◯ |
| xtask | △ | ◯ | - | - | ◯ | - | - | - | - | - | ◯ | - | ◯ | ◯ |
| bench | △ | - | ◯ | - | - | △ | △ | △ | - | - | △ | △ | - | △ |
| ci | △ | ◯ | - | - | - | - | - | △ | - | - | ◯ | ◯ | ◯ | - |

記号: `◯` 主要依存, `△` 間接/運用依存, `-` 依存なしまたは極小

## 6. 生成済み詳細設計ファイル一覧
| Module | Output Path | Status |
|---|---|---|
| `Cargo` | `/mnt/data/libcint_module_designs/cargo_toml_detailed_design.md` | ok |
| `rust_toolchain` | `/mnt/data/libcint_module_designs/rust_toolchain_toml_detailed_design.md` | ok |
| `README` | `/mnt/data/libcint_module_designs/readme_md_detailed_design.md` | ok |
| `LICENSE` | `/mnt/data/libcint_module_designs/license_detailed_design.md` | ok |
| `docs::design::libcint_rust_detailed_design_reviewed` | `/mnt/data/libcint_module_designs/docs_design_libcint_rust_detailed_design_reviewed_md_detailed_design.md` | ok |
| `docs::design::api_manifest` | `/mnt/data/libcint_module_designs/docs_design_api_manifest_csv_detailed_design.md` | ok |
| `docs::design::diagrams` | `/mnt/data/libcint_module_designs/docs_design_diagrams_md_detailed_design.md` | ok |
| `docs::compatibility` | `/mnt/data/libcint_module_designs/docs_compatibility_md_detailed_design.md` | ok |
| `libcint_rs::Cargo` | `/mnt/data/libcint_module_designs/crates_libcint_rs_cargo_toml_detailed_design.md` | ok |
| `libcint_rs::src::lib` | `/mnt/data/libcint_module_designs/crates_libcint_rs_src_lib_rs_detailed_design.md` | ok |
| `libcint_rs::src::api` | `/mnt/data/libcint_module_designs/crates_libcint_rs_src_api_rs_detailed_design.md` | ok |
| `libcint_rs::src::builder` | `/mnt/data/libcint_module_designs/crates_libcint_rs_src_builder_rs_detailed_design.md` | ok |
| `libcint_rs::src::prelude` | `/mnt/data/libcint_module_designs/crates_libcint_rs_src_prelude_rs_detailed_design.md` | ok |
| `libcint_core::Cargo` | `/mnt/data/libcint_module_designs/crates_libcint_core_cargo_toml_detailed_design.md` | ok |
| `libcint_core::src::lib` | `/mnt/data/libcint_module_designs/crates_libcint_core_src_lib_rs_detailed_design.md` | ok |
| `libcint_core::src::atom` | `/mnt/data/libcint_module_designs/crates_libcint_core_src_atom_rs_detailed_design.md` | ok |
| `libcint_core::src::shell` | `/mnt/data/libcint_module_designs/crates_libcint_core_src_shell_rs_detailed_design.md` | ok |
| `libcint_core::src::basis` | `/mnt/data/libcint_module_designs/crates_libcint_core_src_basis_rs_detailed_design.md` | ok |
| `libcint_core::src::env` | `/mnt/data/libcint_module_designs/crates_libcint_core_src_env_rs_detailed_design.md` | ok |
| `libcint_core::src::operator` | `/mnt/data/libcint_module_designs/crates_libcint_core_src_operator_rs_detailed_design.md` | ok |
| `libcint_core::src::tensor` | `/mnt/data/libcint_module_designs/crates_libcint_core_src_tensor_rs_detailed_design.md` | ok |
| `libcint_core::src::error` | `/mnt/data/libcint_module_designs/crates_libcint_core_src_error_rs_detailed_design.md` | ok |
| `libcint_ops::Cargo` | `/mnt/data/libcint_module_designs/crates_libcint_ops_cargo_toml_detailed_design.md` | ok |
| `libcint_ops::build` | `/mnt/data/libcint_module_designs/crates_libcint_ops_build_rs_detailed_design.md` | ok |
| `libcint_ops::src::lib` | `/mnt/data/libcint_module_designs/crates_libcint_ops_src_lib_rs_detailed_design.md` | ok |
| `libcint_ops::src::generated::api_manifest` | `/mnt/data/libcint_module_designs/crates_libcint_ops_src_generated_api_manifest_rs_detailed_design.md` | ok |
| `libcint_ops::src::generated::api_manifest` | `/mnt/data/libcint_module_designs/crates_libcint_ops_src_generated_api_manifest_csv_detailed_design.md` | ok |
| `libcint_ops::src::resolver` | `/mnt/data/libcint_module_designs/crates_libcint_ops_src_resolver_rs_detailed_design.md` | ok |
| `libcint_runtime::Cargo` | `/mnt/data/libcint_module_designs/crates_libcint_runtime_cargo_toml_detailed_design.md` | ok |
| `libcint_runtime::src::lib` | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_lib_rs_detailed_design.md` | ok |
| `libcint_runtime::src::validator` | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_validator_rs_detailed_design.md` | ok |
| `libcint_runtime::src::planner` | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_planner_rs_detailed_design.md` | ok |
| `libcint_runtime::src::scheduler` | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_scheduler_rs_detailed_design.md` | ok |
| `libcint_runtime::src::workspace` | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_workspace_rs_detailed_design.md` | ok |
| `libcint_runtime::src::dispatch` | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_dispatch_rs_detailed_design.md` | ok |
| `libcint_runtime::src::metrics` | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_metrics_rs_detailed_design.md` | ok |
| `libcint_runtime::src::options` | `/mnt/data/libcint_module_designs/crates_libcint_runtime_src_options_rs_detailed_design.md` | ok |
| `libcint_cpu::Cargo` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_cargo_toml_detailed_design.md` | ok |
| `libcint_cpu::src::lib` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_lib_rs_detailed_design.md` | ok |
| `libcint_cpu::src::kernels::one_electron` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_kernels_one_electron_rs_detailed_design.md` | ok |
| `libcint_cpu::src::kernels::two_electron` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_kernels_two_electron_rs_detailed_design.md` | ok |
| `libcint_cpu::src::kernels::center_2c2e` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_kernels_center_2c2e_rs_detailed_design.md` | ok |
| `libcint_cpu::src::kernels::center_3c1e` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_kernels_center_3c1e_rs_detailed_design.md` | ok |
| `libcint_cpu::src::kernels::center_3c2e` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_kernels_center_3c2e_rs_detailed_design.md` | ok |
| `libcint_cpu::src::kernels::center_4c1e` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_kernels_center_4c1e_rs_detailed_design.md` | ok |
| `libcint_cpu::src::transform::c2s` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_transform_c2s_rs_detailed_design.md` | ok |
| `libcint_cpu::src::transform::c2spinor` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_transform_c2spinor_rs_detailed_design.md` | ok |
| `libcint_cpu::src::screening` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_screening_rs_detailed_design.md` | ok |
| `libcint_cpu::src::simd` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_simd_rs_detailed_design.md` | ok |
| `libcint_cpu::src::executor` | `/mnt/data/libcint_module_designs/crates_libcint_cpu_src_executor_rs_detailed_design.md` | ok |
| `libcint_cubecl::Cargo` | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_cargo_toml_detailed_design.md` | ok |
| `libcint_cubecl::src::lib` | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_src_lib_rs_detailed_design.md` | ok |
| `libcint_cubecl::src::executor` | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_src_executor_rs_detailed_design.md` | ok |
| `libcint_cubecl::src::kernels` | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_src_kernels_rs_detailed_design.md` | ok |
| `libcint_cubecl::src::transfer` | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_src_transfer_rs_detailed_design.md` | ok |
| `libcint_cubecl::src::resident_cache` | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_src_resident_cache_rs_detailed_design.md` | ok |
| `libcint_cubecl::src::specialization` | `/mnt/data/libcint_module_designs/crates_libcint_cubecl_src_specialization_rs_detailed_design.md` | ok |
| `libcint_compat::Cargo` | `/mnt/data/libcint_module_designs/crates_libcint_compat_cargo_toml_detailed_design.md` | ok |
| `libcint_compat::src::lib` | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_lib_rs_detailed_design.md` | ok |
| `libcint_compat::src::raw` | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_raw_rs_detailed_design.md` | ok |
| `libcint_compat::src::legacy` | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_legacy_rs_detailed_design.md` | ok |
| `libcint_compat::src::helpers` | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_helpers_rs_detailed_design.md` | ok |
| `libcint_compat::src::optimizer` | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_optimizer_rs_detailed_design.md` | ok |
| `libcint_compat::src::transform` | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_transform_rs_detailed_design.md` | ok |
| `libcint_compat::src::layout` | `/mnt/data/libcint_module_designs/crates_libcint_compat_src_layout_rs_detailed_design.md` | ok |
| `libcint_capi::Cargo` | `/mnt/data/libcint_module_designs/crates_libcint_capi_cargo_toml_detailed_design.md` | ok |
| `libcint_capi::src::lib` | `/mnt/data/libcint_module_designs/crates_libcint_capi_src_lib_rs_detailed_design.md` | ok |
| `libcint_capi::src::errors` | `/mnt/data/libcint_module_designs/crates_libcint_capi_src_errors_rs_detailed_design.md` | ok |
| `libcint_capi::src::shim` | `/mnt/data/libcint_module_designs/crates_libcint_capi_src_shim_rs_detailed_design.md` | ok |
| `libcint_oracle::Cargo` | `/mnt/data/libcint_module_designs/crates_libcint_oracle_cargo_toml_detailed_design.md` | ok |
| `libcint_oracle::build` | `/mnt/data/libcint_module_designs/crates_libcint_oracle_build_rs_detailed_design.md` | ok |
| `libcint_oracle::src::lib` | `/mnt/data/libcint_module_designs/crates_libcint_oracle_src_lib_rs_detailed_design.md` | ok |
| `libcint_oracle::src::compare` | `/mnt/data/libcint_module_designs/crates_libcint_oracle_src_compare_rs_detailed_design.md` | ok |
| `libcint_oracle::src::fixtures` | `/mnt/data/libcint_module_designs/crates_libcint_oracle_src_fixtures_rs_detailed_design.md` | ok |
| `xtask::Cargo` | `/mnt/data/libcint_module_designs/xtask_cargo_toml_detailed_design.md` | ok |
| `xtask::src::main` | `/mnt/data/libcint_module_designs/xtask_src_main_rs_detailed_design.md` | ok |
| `xtask::src::manifest_audit` | `/mnt/data/libcint_module_designs/xtask_src_manifest_audit_rs_detailed_design.md` | ok |
| `xtask::src::oracle_update` | `/mnt/data/libcint_module_designs/xtask_src_oracle_update_rs_detailed_design.md` | ok |
| `xtask::src::gen_docs` | `/mnt/data/libcint_module_designs/xtask_src_gen_docs_rs_detailed_design.md` | ok |
| `xtask::src::bench_report` | `/mnt/data/libcint_module_designs/xtask_src_bench_report_rs_detailed_design.md` | ok |
| `benches::micro_families` | `/mnt/data/libcint_module_designs/benches_micro_families_rs_detailed_design.md` | ok |
| `benches::macro_molecules` | `/mnt/data/libcint_module_designs/benches_macro_molecules_rs_detailed_design.md` | ok |
| `benches::crossover_cpu_gpu` | `/mnt/data/libcint_module_designs/benches_crossover_cpu_gpu_rs_detailed_design.md` | ok |
| `ci::feature_matrix` | `/mnt/data/libcint_module_designs/ci_feature_matrix_yml_detailed_design.md` | ok |
| `ci::oracle_compare` | `/mnt/data/libcint_module_designs/ci_oracle_compare_yml_detailed_design.md` | ok |
| `ci::gpu_bench` | `/mnt/data/libcint_module_designs/ci_gpu_bench_yml_detailed_design.md` | ok |

## 7. 欠落確認
- Source Tree 第15章から抽出した候補は **全件列挙済み**。
- 除外対象も個別メモを生成しており、候補抜けはない。

## 8. 不完全項目一覧
- exact microkernel formula、最終SIMD/tiling閾値、CPU/GPU crossover閾値は上位設計でも実測確定前のため、各対象文書で「要追加調査」と明記。
- generated artifact の列順・schema minor revision は `compiled_manifest.lock.json` 更新時に同時更新が必要。
- 4c1e の適用範囲は `Validated4C1E` に限定し、拡張時は oracle/identity/property test の追加が前提。
- F12/STG/YP は現時点で sph-only。cart/spinor は未実装ではなく out-of-scope として扱う。

## 9. stub ファイル一覧
- なし
