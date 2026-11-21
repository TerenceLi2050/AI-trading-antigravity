# 中证 800 量化选股

默认入口自行取得中证800当期成分、交易日历、260个交易日的行情和已披露财报，完成校验后发布两份各20只的质量候选名单。开发回测另有入口。当前进度、数据审计结论和后续验收统一见[《中证800量化选股系统 · 最终情况汇报与后续任务规划》](中证800项目进展与交接.md)。

## 运行

当前项目可直接运行。新机器使用 Python 3.11 或更新版本，日常包安装命令为 `python -m pip install .`。需要模型、回测和测试时安装 `python -m pip install ".[research,test]"`。Windows x64 的 Python 3.12 可先运行 `python -m pip install --only-binary=:all: -r requirements-win-py312.lock` 复现已核验的版本。

也可以将[已验证的程序包](dist/csi800_quant_screen-0.3.0-py3-none-any.whl)复制到另一台机器，在文件所在目录运行 `python -m pip install csi800_quant_screen-0.3.0-py3-none-any.whl`，再运行 `quant-screen --root "D:\QuantData"`。行情、财报和报告会落在所选目录。

```powershell
python run_daily_screen.py --doctor
python run_daily_screen.py --build-history
python run_daily_screen.py --build-history --assemble-local
python run_daily_screen.py --mode all --standalone
python run_daily_screen.py
python run_daily_screen.py --model reports/runs/<研究批次>/model.joblib
python run_daily_screen.py --watch
python run_daily_screen.py --mode screen
python run_daily_screen.py --mode all
python run_daily_screen.py --audit-universe
python run_daily_screen.py --collect-universe-prices
python -m pytest -q
powershell -ExecutionPolicy Bypass -File scripts/manage_windows_task.ps1 -Action install
```

- `--doctor` 一次性检查依赖版本、BaoStock 退避状态、日常与历史规范库就绪度以及报告指针有效性。
- `--build-history` **无需任何旧 CSV**，从空目录自动构建独立多年历史研究库：`sessions.csv`（交易日历，BaoStock 主源 + AkShare 备源）、`calendar.csv`（000906 价格指数）、`csi800_total_return.csv`（H00906 全收益指数）、`membership.csv`（动态历史成分与行业半开区间）、`quotes.parquet`（全量并集多年原始+后复权日线，支持 `--bar-provider auto|baostock|akshare` 断点续传）以及 `quality_history.csv`（东财按季度 PIT 财报）。可用 `--assemble-local` 直接把本机已归档分片拼装成独立 `data/input/` 规范库，或用 `--skip-bars` / `--skip-financials` 分步运行。
- `--mode all --standalone` 强制仅读取 `data/input/quotes.parquet` 等独立规范表运行完整滚动训练、撮合回测、实际发布五组规则（含行业分散版）历史对照和分年度绩效统计，完全不依赖旧 CSV。
- `--model <model.joblib>`（或在 `configs/default.toml` 的 `[screening]` 中配置 `model_path` 与 `model_weight`）可将验证通过的冻结模型接入日常选股打分，并在报告中输出真实特征归因贡献。
- `scripts/manage_windows_task.ps1` 支持在 Windows 计划任务中一键注册/查询/卸载工作日盘后自动选股任务；`--watch` 同时写入 `reports/watch.log` 与 `reports/watch_status.json` 心跳。

- 默认执行 `daily`。首次运行需能连接中证指数、BaoStock 和东方财富。按股票保存下载进度，后续交易日校对重叠价格并增量更新。同日已验收的批次可直接复用。用 `--refresh` 可再次采集。安装后可在任意路径运行 `quant-screen --root "D:\QuantData"`，程序随包携带默认配置。
- 如已有其他机器完整采集的归档，可运行 `python run_daily_screen.py --import-release "D:\归档\批次目录"`。程序在本机核对各文件摘要，按原始采集时间和交易日历离线复算。导入后的数据同样接受信号日和收盘时效检查。
- BaoStock 的日常和历史采集共用本机单连接并限制请求频率。收到明确的访问拒绝后，会立即保留进度，至少24小时后才容许本机再次请求。24小时是程序的重试间隔，接口何时恢复由数据源决定。
- `--watch` 启动后先执行一次，再按北京时间和交易日历等待下一次收盘。默认从15:35检查数据，遇到发布延迟每30分钟重试，按 Ctrl+C 停止。首次网络下载和每次执行均自动完成，无逐股审批步骤。
- `screen` 复算已归档的日常规则。已有旧工程且尚无独立日常批次时，可离线复用旧行情。默认 `daily` 从独立来源开始。两者的日常排名均无需训练模型。
- `all` 使用本地历史数据训练 Ridge、LightGBM 和 CatBoost，进行开发期滚动评估，结果指针为 `reports/latest_research.json`。历史研究可使用旧 CSV 或通过校验的转换缓存。
- `--as-of YYYY-MM-DD` 只读取在所指定北京时间日期结束前已经归档的独立批次。`--replay-run 运行编号` 精确复算某次独立日常报告，校验当时的输入、程序、配置和两份名单的代码顺序及分数摘要。回放结果写入 `latest_replay.json`。
- 本期11月20日收盘数据于11月21日归档。复算本期两版名单可直接执行 `python run_daily_screen.py --replay-run 20261121T055423Z-5d2dadf2`。
- `--audit-universe` 输出开发期完整成分抽样的逐期覆盖、缺失证券和已补历史行情的动量排名敏感性。`--collect-universe-prices` 从公开源补采对应证券的原始价、后复权价和交易状态。补采档案位于 `data/historical_observations/development_bars/`。
- `audit` 用于旧研究输入审计，`--probe` 检查公开源。新批次缺失、损坏或过期时，日常入口寻找可校验的归档，重新核查日期和门槛。退出码0表示通过日常发布或回放，2表示数据不足或降级。详情在该次运行的 `screen.json` 和 `refresh.json`。
- `--refresh-calendar` 单独更新独立日历，`--collect-history` 归档公开成分抽样和中证全收益基准，`--collect-raw-history` 回补截至开发边界前的旧股票池原始行情及交易状态。补采批次按证券保存进度。

每次运行生成独立目录 `reports/runs/`。`reports/latest.json` 指向通过交易日、收盘采集和基本面门槛的最近正式报告，`latest_run.json` 指向最新一次筛选尝试。休市或当日收盘确认前使用最近完整交易日。默认收盘采集截止门槛是北京时间 15:35。行情落后于应有交易日、缺少当天收盘后采集证据或日历未覆盖时，报告会降级。`report.html` 可直接打开。 [沪市交易规则](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml)

| 输出 | 含义 |
|---|---|
| `quality_screen.csv` | 成分、行情时效、交易状态和财务覆盖达到门槛后的质量名单，保留名称、财报期、公告日、分数和风险标记 |
| `quality_screen_diversified.csv` | 同一候选池的行业分散版。默认单行业至多 5 只，金融相关行业合计至多 7 只 |
| `research_watchlist.csv` | 任一数据门槛未通过时输出的观察名单，每行标记 `RESEARCH_ONLY` |
| `audit.json`、`coverage.csv`、`screen.json`、`screening_universe.csv` | 文件和代码校验值、数据覆盖、来源、数据更新状态和逐股门槛记录 |
| `research.json`、`signal_diagnostics.csv`、`factor_diagnostics.csv`、`factor_correlations.csv`、`permutation_diagnostics.csv` | 费后绩效、隔离后的训练窗、RankIC、HAC t 值、五分组收益、因子相关性和 LambdaRank 样本外截面置乱 |
| `purged_cv.csv`、`portfolio_nav.csv`、`fills.csv` | 净化组合切分诊断、逐日净值和模拟成交 |
| `model.json`、`model.joblib` | 当次训练元数据和本地冻结模型 |
| `cpcv_paths.csv`、`cpcv_path_nav.csv` | 6 块两两留出组装成 5 条完整 Ridge 路径的诊断净值与指标 |
| `quality_research.json` | 财务综合策略的开发对照或阻塞理由。数据齐全后比较质量、量价、混合和模型在同一股票池上的表现 |
| `experiment_manifest.json`、`factor_family_ablation.csv` | 当批已登记规则、模型配置、敏感性分析及 Ridge 因子族消融 |
| `prospective_outcomes.csv` | 已归档的正式日常名单按实际观察时间记录信号，逐次检查计划入场、标签到期及收益 |
| `prospective_cohorts.csv`、`prospective_summary.csv` | 同时登记综合原榜、分散榜、纯财务、财务筛后量价和全池量价，每版独立统计可实现收益与未到期数量。持有期在登记时固化 |
| `data/daily/<批次>/manifest.json` | 全部输入文件的校验值、采集时间、行情区间、来源和完整度。每一批可独立复算当期截面 |

## 质量筛选和研究口径

- **数据治理。** 旧 CSV 通过字段位置白名单投影为 Parquet，原件和 Notebook 保留。旧财务、估值和历史行业快照不会进入模型。旧开盘价由复权比率还原并按 VWAP 复核，历史停牌和风险警示状态的缺失单独统计。增量更新按股票代码与交易日配对原始和后复权行情，检查重叠价格和复权尺度，达到覆盖率后发布带哈希的版本。
- **当期初筛。** 用同日中证800成分、原始行情和ST状态过滤。独立日常入口采用同报告期、已经披露的年内累计财务，对公告、版本日期和报表账龄逐股检查。亏损、非正净资产及非金融公司的低现金转化均有记录，金融企业在适用的盈利和成长维度内打分。默认财务质量分占65%，量价规则分占35%。旧工程的研究路径另支持按公告生效的TTM财务。多年历史检验需要逐期财务版本和成分生效记录。
- **信号和模型。** 17 个量价特征使用同日横截面百分位。下一交易日开盘入场，20 个交易日后的开盘退出。冻结、停牌或缺价将严格标签标为缺失，另外统计顺延退出的敏感性。用实际退出日和 5 日隔离带决定训练范围。Ridge、LightGBM LambdaRank 和 Huber、CatBoost 及其等权百分位集成均与规则基线比较。TreeSHAP 显示 LambdaRank 原始分数贡献。
- **组合。** 默认每 20 个交易日调仓，Top 20 和 Top 40 缓冲区，报告等权、逆波动、Ledoit–Wolf HRP 和双倍成本。开盘价与滑点成本分别入账。开盘可执行判定使用当日开盘、交易状态、官方限价或有效前收，旧样本缺参考时用前一交易日收盘保守推算。冻结仓位仍占用实际名额，公司行为继续用复权持仓近似。 [上交所交易规则](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml) · [深交所风险警示板指南](https://www.szse.cn/lawrules/service/member/t20260630_621404.html) · [印花税减半公告](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html)
- **验收。** 锁箱从 2025-06-01 起。开发数据、DSR、PBO 和成本约束通过后，可用 `--mode all --lockbox freeze` 固定窗口、模型、输入与验收标准。`--mode all --lockbox evaluate` 校验该协议后才能一次性评估。DSR 和 PBO 使用本批固定信号与非重叠调仓区间，同时记录策略相关性和区块重采样区间。 [中证 800 指数资料](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000906factsheet.pdf)

## 可选数据输入

文件放到 `data/input/`。股票代码用六位字符串，日期用 `YYYY-MM-DD`。按日期和代码拼接，重复键、非法日期、相互矛盾的值会拒绝入库。经营与财务金额用人民币元，缺失金额留空。

| 输入 | 主要列和时间规则 |
|---|---|
| `membership.csv` | `code,start_date,end_date,ann_date,industry,source`。`start_date` 含当日，`end_date` 不含当日，仍生效时留空，公告应早于生效日。需要官方历史生效区间。 |
| `financials.csv` | `code,report_date,ann_date,source,version,revenue_ytd,net_income_ytd,cfo_ytd,gross_profit_ytd,assets,equity,liabilities`。每行一个公告版本，公告日严格早于信号日。累计损益用上年年报加本期减上年同季计算 TTM。 |
| `status.csv` | `date,code,trade_status,is_st,limit_up,limit_down,source`。0/1 分别代表停牌/可交易与非 ST/ST，限价为原始价。 |
| `calendar.csv` | `date,market_open,market_close,source`。可以补充全部交易日的 `total_return_open,total_return_close`。全收益不完整时整段使用价格指数口径。 |
| `sessions.csv` | `date,is_session,source`。每个自然日一行，休市日为 0，用于确认应有的收盘交易日。 |
| `csi800_total_return.csv` | `date,total_return_close,source`，可选 `total_return_open`。中证公开源实际提供 H00906 收盘，组合使用全收益收盘，开盘标签使用有开盘值的 000906 价格指数。 |
| `raw_history.parquet` | 对旧股票池的 `date,code,raw_open,raw_high,raw_low,raw_close,raw_preclose,trade_status,is_st` 核验补充。旧复权数据按匹配阈值保留用于收益口径。 |
| `quality_history.csv` | `code,report_date,ann_date,version_known_date,source,version,roe_ytd_pct,book_per_share,profit_ytd,eps_ytd,operating_cash_per_share_ytd,profit_growth_pct,revenue_growth_pct`。版本真实可见日须早于信号日。 |
| `quotes.parquet` | `date,code,raw_open,raw_high,raw_low,raw_close,adj_open,adj_high,adj_low,adj_close,volume,amount,turnover,vendor_pct,trade_status,is_st,price_source`。成交量单位股、换手率为小数、日涨跌幅为百分点。可选 `total_shares,limit_up,limit_down`。 |

复核参考原始审计文件 `中证800数据采集与质量审计.md` 和交接记录。BaoStock 沪深 300 与中证 500 的历史月份观察保存在 `data/historical_observations/`。它用于核对股票池缺口，正式研究需输入连续、可核验的成分生效区间。财报版本、公司行为现金流和官方开盘竞价数据仍需按相应口径接入。
