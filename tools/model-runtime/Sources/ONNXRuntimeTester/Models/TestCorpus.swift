import Foundation

/// 内置测试语料：加载模型后自动生成一批测试数据
enum TestCorpus {

    /// Embedding 测速用句子（中英文混合，长短不一）
    static let embeddingTexts: [String] = [
        "杭州是一座融合自然山水与传统文化的城市，西湖是其最著名的地标。",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "机器学习模型推理速度通常以每秒生成的 token 数量来衡量。",
        "Apple Silicon 的统一内存架构让 CPU 与 GPU 可以零拷贝共享数据。",
        "ONNX Runtime 是一个跨平台的推理引擎，支持多种执行提供程序。",
        "北京故宫是中国明清两代的皇家宫殿，也是世界上现存规模最大的木质结构古建筑群。",
        "Vector databases store embeddings for similarity search at scale.",
        "embedding 模型可以把任意文本映射成一个固定维度的稠密向量。",
        "上海的东方明珠广播电视塔是浦东新区最具代表性的建筑之一。",
        "The transformer architecture revolutionized natural language processing.",
        " reranker 模型用于对候选文档进行相关性重排序。",
        "深圳是中国改革开放后设立的第一个经济特区，如今已成为科技创新之都。",
        "Core ML 执行提供程序可以把部分计算子图卸载到苹果的神经引擎上。",
        "Swift 是一门强调安全与性能的现代编程语言，适合开发系统级应用。",
        "广州塔又被称作小蛮腰，是广州城市新中轴线上的标志性建筑。",
        "Quantization reduces model size by representing weights with fewer bits.",
        "苏州园林以拙政园和留园最为著名，体现了江南古典园林的艺术精髓。",
        "Distributed training requires careful synchronization of gradients.",
        "西安是十三朝古都，兵马俑和古城墙吸引了来自世界各地的游客。",
        "The attention mechanism allows models to focus on relevant parts of the input.",
        "南京夫子庙秦淮河一带是六朝烟月之区，金粉荟萃之所。",
        "Batch inference can significantly improve throughput on modern accelerators.",
        "成都以大熊猫和火锅闻名，也是一座来了就不想离开的城市。",
        "Memory bandwidth is often the bottleneck in large language model inference.",
        "武汉黄鹤楼与岳阳楼、滕王阁并称为江南三大名楼。",
        "Cosine similarity measures the angle between two non-zero vectors.",
        "重庆是一座山城，长江与嘉陵江在此交汇，夜景非常壮观。",
        "Edge deployment of neural networks demands efficient model compression.",
        "天津狗不理包子和十八街麻花是当地著名的传统小吃。",
        "Tokenization splits raw text into the subword units a model understands.",
        "长沙岳麓山下聚集了湖南大学和中南大学等众多高校。",
        "Neural engines specialize in low-precision matrix multiplication workloads.",
        "青岛红瓦绿树、碧海蓝天，德式建筑群遍布老城区。",
        "Speculative decoding uses a small draft model to propose candidate tokens.",
        "昆明四季如春，滇池边每年冬天都会聚集大量红嘴鸥。",
        "Inference latency depends on model size, hardware, and batch composition.",
        "厦门鼓浪屿保留了众多中西合璧的历史建筑，禁止机动车通行。",
        "Mean pooling averages token embeddings to produce a sentence embedding.",
        "哈尔滨冰雪大世界每年冬天都会用冰雕再现世界著名建筑。",
        "Prefill and decode are the two distinct phases of autoregressive generation.",
        "拉萨布达拉宫屹立在红山之上，是藏传佛教的圣地。",
    ]

    /// Reranker 效果演示用的 query
    static let rerankerQueries: [String] = [
        "杭州有什么著名的旅游景点？",
        "如何提升模型推理速度？",
        "上海有哪些标志性建筑？",
        "什么是 embedding 向量？",
    ]

    /// 每个 query 对应的候选文档（第 0 条为相关文档，其余为干扰项）
    static let rerankerDocuments: [[String]] = [
        [
            "杭州西湖是世界文化遗产，十景闻名中外，是杭州最著名的旅游景点。",
            "北京故宫是中国明清两代的皇家宫殿，现为故宫博物院。",
            "Python 是一种广泛使用的解释型编程语言。",
            "这家餐厅的菜品味道一般，但是价格很实惠。",
            "昨天的足球比赛非常精彩，主队最终以二比一获胜。",
        ],
        [
            "使用 GPU 或神经引擎加速、批处理请求、量化模型权重都能提升推理速度。",
            "杭州西湖的断桥残雪是西湖十景之一。",
            "定期清理手机缓存可以让系统运行更流畅。",
            "他每天早上都会去公园跑步锻炼身体。",
            "这家书店的老板收藏了许多绝版图书。",
        ],
        [
            "上海东方明珠广播电视塔和外滩万国建筑博览群都是上海的标志性建筑。",
            "成都的大熊猫繁育研究基地每年吸引大量游客。",
            " quantized models can run faster on neural engines.",
            "她喜欢在周末去图书馆看小说。",
            "新买的洗衣机噪音有点大，需要退货。",
        ],
        [
            "embedding 向量是把文本映射到高维空间后得到的稠密数值表示，可用于相似度检索。",
            "苏州园林的造园艺术体现了江南文人的审美情趣。",
            "这款手机的电池续航能力非常出色。",
            "公司下周要召开年度总结会议。",
            "高铁从北京到上海只需要四个半小时。",
        ],
    ]

    /// Embedding 效果演示：同主题（应高相似度）与不同主题（应低相似度）
    static let similarityPairs: [(label: String, a: String, b: String)] = [
        ("同主题", "杭州西湖是著名的旅游景点。", "西湖位于杭州，是最受欢迎的景区之一。"),
        ("同主题", "机器学习需要大量训练数据。", "训练神经网络通常要准备足够多的样本。"),
        ("不同主题", "杭州西湖是著名的旅游景点。", "高通量筛选技术在药物研发中应用广泛。"),
        ("不同主题", "机器学习需要大量训练数据。", "昨晚的烤鱼味道非常好，推荐一试。"),
    ]

    /// 生成 count 条测试文本（语料循环 + 序号后缀保证唯一）
    static func generateTexts(count: Int) -> [String] {
        (0..<count).map { index in
            let base = embeddingTexts[index % embeddingTexts.count]
            return "\(base)（样本 \(index + 1)）"
        }
    }
}
