# 第 18 章学习笔记：用密码学回执保护 AI Agents

> 核心一句话：**给 agent 的每一次动作发一张"带签名的收据"（receipt），让外部审计员不用信任你、只凭公钥就能验证"这事是谁干的、内容没被改、先后顺序如何"——但收据只证明归属/完整/顺序，不证明动作本身正确。**

---

## 1. 最该记住的几个点

- **审计追踪的根本难题：日志不可信**。普通日志是 agent 自己写的、谁有文件系统权限谁就能改；云日志只"平台级防篡改"，前提是审计员信任云平台。受监管场景（金融/医疗/EU AI Act）里这种信任不成立。**密码学回执的解法：让每条动作独立可验证——审计员不需要信你，只需你的公钥 + 回执本身。**
- **回执 = 一张被签名的 JSON 记录**。它干三件事：①**签名**——agent 网关用 Ed25519 私钥签，任何人拿对应公钥可离线验证，改任意字段签名即废；②**规范编码（canonicalization，JCS / RFC 8785）**——签名前先把 JSON 序列化成字节级确定的形式，否则不同序列化器对同一内容会出不同签名；③**哈希链（hash chaining）**——每张回执存上一张的 `previous_receipt_hash`，删/改/ reorder 中间一张，后面全断。
- **回执只给三个保证，且只有三个**：**归属（Attribution）**——这个密钥签了这段内容；**完整（Integrity）**——签名后内容没变过；**顺序（Ordering）**——这张在链上排在那张之后。其余一概不保证。
- **回执明确"不证明"的才是本课精髓**：①**正确性**——错的回答照样能干净签名；②**策略合规**——`policy_id` 记录的只是"声称用了某策略"，不是"策略真被评估过/允许了"；③**密钥之外的人**——回执说"这密钥签的"，不说"这人授权的"，把密钥关联到具体人要另建身份基础设施；④**输入真实性**——agent 收到被篡改的 prompt 并照做，回执照样忠实记录。结论：**"我们有回执"≠"我们被治理了"**，回执只是地基，治理是你在上面搭的系统。
- **人类批准回执 = 同套原语的延伸**。普通动作回执只说"密钥签了"，高风险动作（退款/删除/转账）要求的是"人类授权了这确切动作"。做法是加一种 `human.approval.v1` 回执：具名批准人在**执行前**对"完整规范动作 + 其摘要"签名；agent 执行后的动作回执带同一动作摘要 + `parent_approval_ref`（指向批准回执的 `receipt_hash`）。`verify_chain` 用**两套各自钉死的密钥库**（批准人 vs agent）校验——代码路径共享，但权限永不混用。
- **关键边界：签名批准 ≠ 权力本身**。批准只有在"执行时两张回执仍绑定同一规范动作"才构成有效权力。笔记里的拒绝用例让这点变真：stale authority（策略版本变了/批准密钥被轮换/批准过期，签名仍有效也被拒）、digest substitution（动作回执指向一个真实但绑定**不同**规范动作的批准，被拒）。

## 2. 回执是怎么造出来的（核心步骤）

一张回执的本质，是把"agent 调了哪个工具、参数与结果"记作 JSON，再用 Ed25519 签名。就三步，纯 Python 几行搞定，无需特殊库：

1. **造 payload（先不签名）**：下面是一张 `agent.tool_call.v1` 类型回执的**典型字段示例**——`type` / `agent_id` / `tool_name` / `tool_args_hash` / `result_hash` / `policy_id` / `timestamp` / `sequence` / `previous_receipt_hash`。注意这些都是**按业务自定义的，并非固定 schema**：你可以增删字段（如加 `request_id` 做链路追踪），甚至定义全新 `type`（如后续的 `human.approval.v1` 字段就完全不同）。唯一硬约束是——**`signature` 必须放在 payload 之外、且签名覆盖除它以外的全部字段**（用 JCS 规范编码）；若要连成防篡改链，则保留 `previous_receipt_hash`。参数和结果存的是 **SHA-256 哈希**而非原文——既保护隐私（PII/业务数据不进回执），又让回执体积恒定。
   ```python
   payload = {
       "type": "agent.tool_call.v1",
       "agent_id": "contoso-travel-bot",
       "tool_name": "lookup_flights",
       "tool_args_hash": sha256_canonical(tool_args),   # 参数的摘要
       "result_hash":   sha256_canonical(tool_result),  # 结果的摘要
       "policy_id": "contoso-travel-policy-v3",
       "timestamp": "2026-04-25T14:30:00Z",
       "sequence": 0,
       "previous_receipt_hash": None,   # 链上第一张为 None，之后指向前一张哈希
   }
   ```
2. **规范编码 + 哈希 + 签名**：用 JCS（`jcs.canonicalize`）转成确定字节 → SHA-256 → Ed25519 私钥签名 → 把签名作为独立 `signature` 对象（含 `alg` / `sig` / `public_key`）**挂在 payload 之外**（签名不进被签字节，避免自引用）。

   > **说明：Ed25519 私钥 vs 平时常用的 RSA 私钥。** ①**算法家族不同**：Ed25519 属**椭圆曲线**签名（EdDSA，曲线 Curve25519，对应 RFC 8032），而 RSA 是**大整数分解**体系——两者数学基础完全不同。②**密钥短得多**：Ed25519 私钥固定 32 字节、公钥也 32 字节、签名 64 字节；RSA 要达到 comparable 安全级（约 128-bit）需 3072 位（384 字节）密钥、签名更长。回执场景里"小"直接省存储/带宽（本章每张回执才 ~500 字节，RSA 会显著变胖）。③**更快**：Ed25519 签名/验签比 RSA 快一个数量级，且验签无需网络、纯本地。④**设计上不易误用**：Ed25519 是"确定性签名"（不依赖随机数生成器，避免 RSA/ECDSA 因弱随机数导致私钥泄露的经典坑），且天然抗长度扩展等。⑤**本课为什么用它**：回执要"每张小、验签快、可离线、实现简单"——Ed25519 全中；RSA 不是不能用，只是又大又慢、在此处无收益。注意 Ed25519 **不是量子安全**的，草案预留了 `signature.alg` 字段以便未来切到后量子标准（如 `ML-DSA-65`），见 §3 进阶模式。
   ```python
   canonical_bytes = canonicalize(payload)
   message_hash    = hashlib.sha256(canonical_bytes).digest()
   signature_bytes = signing_key.sign(message_hash).signature
   receipt = {**payload, "signature": {"alg": "EdDSA",
                                      "sig": b64url_nopad(signature_bytes),
                                      "public_key": b64url_nopad(bytes(verify_key))}}
   ```
3. **连成链**：下一张回执把上一张的哈希写进自己的 `previous_receipt_hash`。删中间一张，后面的 `previous_receipt_hash` 全对不上——攻击者要掩盖，必须拿私钥重签其后所有回执。

   > **说明：`previous_receipt_hash` 到底哈希了什么？** 它不是"把上一张回执原样（含 `signature`）序列化后 hash"，而是按草案规定、对**前驱回执做规范化后取 SHA-256**：典型实现是对前驱回执中**被签名的那段（`signature` 之外的 payload，含前驱自己的 `previous_receipt_hash`）**做 JCS 编码再哈希（也有的实现直接对整张回执取 `receipt_hash` 引用，口径差异不影响结论）。无论含不含 `signature`，防篡改效果一样——前一张 payload 任一字节被改 → 它自己的签名立即作废 → 下一张的 `previous_receipt_hash` 对不上 → 链断。所以纠结"是否含 signature"只是哈希对象的定义口径问题，**核心机制是：每张回执用哈希指死前一张的内容，改任一张都得拿私钥重签其后所有回执**。

**验证是其逆操作**：取出 `signature` 外的 payload → 重新 JCS 编码 + SHA-256 → 用 `public_key` 验签。全程离线、无网络、无第三方信任。改一个字节（如把 `policy_id` 改成更宽松的策略）即验签失败。

> 底层标准：Ed25519（RFC 8032）+ JCS（RFC 8785）+ SHA-256，全是广泛部署的成熟原语；本课回执格式遵循 IETF 草案 `draft-farley-acta-signed-receipts`。

## 3. 关键事实速查

- **验签只需公钥**：审计员拿公钥 + 回执即可离线验证，这是回执在"气隙/跨组织/低信任"场景有用的根本原因。
- **哈希而非原文**：`tool_args_hash` / `result_hash` 存摘要，是为了隐私 + 体积恒定；审计员用摘要比对另行存储的真实内容。
- **哈希链的断裂范围**：删中间第 N 张 → 第 N+1 张及之后**全部**失效，因它们的 `previous_receipt_hash` 不再指向真实前驱。
- **回执不证明正确性/合规/身份/输入真实**：这是全课最重要的边界，见 §1。
- **生产化清单（不是教学代码就够了）**：① 私钥迁出开发机（Key Vault / HSM，绝不进源码或明文）；② 公开验证公钥（如 JWK Set 放在 `.well-known/agent-keys.json`）；③ 定期把链头哈希锚到透明日志（Sigstore Rekor / RFC 3161 时间戳）证"此链此时存在"；④ 回执不可变存储（append-only / Object Lock）防内鬼改写历史；⑤ 规划留存（每张 ~500 字节，日 1 万次调用年约 1.8 GB）；⑥ 文档写明回执不覆盖哪些控制（输入校验/策略执行/限流/身份）。
- **自写 vs 用库**：本课走"自写 50 行"路线逼你懂每个原语；生产可上库（如 `protect-mcp` / `@veritasacta/verify` 包 MCP 的 Node 实现、Python SDK `nobulex`、微软 Agent Governance Toolkit 把回执和 Cedar 策略决策组合）。类比：自写 JWT 库 vs 用成熟库，皆可，库省时且减审计面。
- **进阶模式（治理成熟后）**：选择性披露（Merkle 承诺，RFC 6962，对 GDPR 友好）、回执撤销（密钥泄露后标记失效）、双签/拆分签名（执行前授权 + 执行后结果分签）、载荷组合（把决策前推理也封进 `result_hash`）、跨实现一致性（多语言对共享测试向量互验）、后量子迁移（`signature.alg` 可换 `ML-DSA-65`，过渡期双签）。

## 4. 与前一章的对照（安全是本地 agent 之外的另一维度）

| 维度 | 第 17 章（本地 agent） | 第 18 章（回执安全） |
| --- | --- | --- |
| 关注点 | 把模型/工具搬回本机（隐私/成本/离线） | 让 agent 行为可被外部审计、防篡改 |
| 信任假设 | 本机即边界，沙箱限工具范围 | 默认审计员**不信任你**，只信公钥 |
| 核心技术 | OpenAI 兼容端点 + tool-calling 循环 + 沙箱 | Ed25519 签名 + JCS 规范编码 + 哈希链 |
| 落地产物 | 能跑的本地 agent | 每动作一张可验证收据 + 验证链 |

> 细节、完整 notebook（`18-signed-receipts.ipynb` / `human-authorization-receipts.ipynb`）、Knowledge Check 答案 → 回看原 README（`18-securing-ai-agents/README.md`）。
