# 导入必要的库
import os  # 用于文件路径处理、环境变量设置等系统操作
import pandas as pd  # 用于Excel/CSV数据读取、处理和转换
import numpy as np  # 用于数值计算和缺失值处理
# 从swift库导入大模型相关工具：模型加载、数据集处理、模板配置、编码预处理等
from swift.utils import get_logger

try:
    from swift.llm import get_model_tokenizer, get_template, EncodePreprocessor
    try:
        from swift.llm import load_dataset
    except Exception:
        load_dataset = None
    from swift.utils import find_all_linears, get_model_parameter_info, plot_images, seed_everything
    from swift.tuners import Swift, LoraConfig
    from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    SWIFT_TRAIN_AVAILABLE = True
except Exception as e:
    print("[WARN] swift训练相关API导入失败，当前仅运行异常分析模式:", repr(e))
    get_model_tokenizer = None
    load_dataset = None
    get_template = None
    EncodePreprocessor = None
    find_all_linears = None
    get_model_parameter_info = None
    plot_images = None
    seed_everything = None
    Swift = None
    LoraConfig = None
    Seq2SeqTrainer = None
    Seq2SeqTrainingArguments = None
    SWIFT_TRAIN_AVAILABLE = False

# 全局常量与配置
# 设置仅使用第0块GPU（若有多个GPU，可指定其他编号，如'0,1'）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 获取日志器（用于打印程序运行信息、错误提示等，方便调试和跟踪流程）
logger = get_logger()

# 异常检测阈值（根据业务规则定义，用于判断电池数据是否异常）
# 充电/放电共通阈值（两种状态下都需要检查的条件）
MONO_VOLTAGE_JUMP_THRESHOLD = 0.05  # 单体电池电压跳变阈值（单位：V）：相邻两行的电压差超过此值则异常
CURRENT_JUMP_THRESHOLD = 0.5  # 电流跳变阈值（单位：A）：仅充电时检查相邻行电流差
TEMPERATURE_JUMP_THRESHOLD = 3.0  # 温度跳变阈值（单位：℃）：相邻行温度差超过此值则异常

# 充电专属阈值（仅在充电状态下生效的条件）
CHARGE_TOTAL_VOLTAGE_JUMP = 3.0  # 充电时总电压跳变阈值（单位：V）

# 放电专属阈值（仅在放电状态下生效的条件）
DISCHARGE_MAX_TOTAL_VOLTAGE = 378.2  # 放电时总电压最大阈值（单位：V）：超过此值则异常

# 状态编码映射（根据实际业务定义的状态标识，用于区分充电/放电状态）
DISCHARGE_STATE = 30  # 30表示放电状态
CHARGE_STATE = 110  # 110表示充电状态


def detect_anomalies(df):
    """
    检测电池数据中的异常，返回每行数据的异常标记和原因
    逻辑：逐行对比当前行与前一行数据，根据充电/放电状态应用不同的阈值规则

    参数：
        df: pandas.DataFrame，包含电池测试数据（如总电压、电流、单体电压、温度等）
    返回：
        anomalies: 列表，每个元素为字典，包含'is_anomaly'（是否异常）和'reasons'（异常原因）
    """
    anomalies = []  # 存储所有行的异常结果
    prev_row = None  # 记录上一行数据，用于相邻行的跳变对比

    # 提取所有单体电池电压列（适配1号到91号电池，列名包含"号电池单体电压"）
    mono_voltage_cols = [col for col in df.columns if '号电池单体电压' in col]
    logger.info(f"检测到单体电池电压列: {len(mono_voltage_cols)} 列（用于单体电压异常判断）")

    logger.info(f"开始检测异常数据，总行数: {len(df)}")

    # 逐行遍历数据，检查异常
    for index, row in df.iterrows():
        # 每处理100行打印一次进度，方便跟踪
        if index % 100 == 0:
            logger.info(f"正在处理第 {index}/{len(df)} 行数据")

        anomaly = False  # 标记当前行是否异常
        reasons = []  # 存储当前行的异常原因

        # 从当前行提取关键参数（根据实际数据集的列名适配）
        current_state = row.get('整车State状态（状态机编码）', np.nan)  # 状态编码（判断充电/放电）
        total_voltage = row.get('动力电池内部总电压V1', np.nan)  # 总电压（V）
        current = row.get('动力电池充/放电电流', np.nan)  # 电流（A）
        # 单体电压：存储为{列名: 电压值}的字典（过滤空值）
        mono_voltages = {col: row[col] for col in mono_voltage_cols if pd.notna(row[col])}
        # 温度：取1号温度检测点（可根据需求调整为其他检测点）
        current_temp = row.get('1号温度检测点温度', np.nan)

        # 检查关键参数是否为空（空值可能导致异常判断失效）
        state_is_nan = pd.isna(current_state)
        total_voltage_is_nan = pd.isna(total_voltage)
        current_is_nan = pd.isna(current)
        temp_is_nan = pd.isna(current_temp)

        # 根据状态编码判断当前是充电还是放电状态（非空时才判断）
        is_charge = False
        is_discharge = False
        if not state_is_nan:
            is_charge = (current_state == CHARGE_STATE)  # 状态编码=110 → 充电
            is_discharge = (current_state == DISCHARGE_STATE)  # 状态编码=30 → 放电

        # 1. 总电压异常判断（分充电/放电状态处理）
        if not total_voltage_is_nan:  # 总电压非空时才检查
            # 1.1 充电时：检查总电压跳变（与前一行的差值）
            if is_charge and prev_row is not None:  # 充电状态且存在前一行数据
                prev_total_voltage = prev_row.get('动力电池内部总电压V1', np.nan)  # 前一行总电压
                if not pd.isna(prev_total_voltage):  # 前一行总电压非空
                    voltage_jump = abs(total_voltage - prev_total_voltage)  # 计算电压跳变值
                    if voltage_jump > CHARGE_TOTAL_VOLTAGE_JUMP:  # 超过充电总电压跳变阈值
                        anomaly = True
                        reasons.append(
                            f"充电时总电压跳变过大（上一行{prev_total_voltage:.2f}V→本行{total_voltage:.2f}V，"
                            f"跳变{voltage_jump:.2f}V > 阈值{CHARGE_TOTAL_VOLTAGE_JUMP}V）"
                        )

            # 1.2 放电时：检查总电压是否超过最大阈值
            if is_discharge:  # 放电状态
                if total_voltage > DISCHARGE_MAX_TOTAL_VOLTAGE:  # 超过放电总电压上限
                    anomaly = True
                    reasons.append(
                        f"放电时总电压超过最大阈值（{total_voltage:.2f}V > 阈值{DISCHARGE_MAX_TOTAL_VOLTAGE}V）"
                    )

        # 2. 单体电池电压异常判断（充电/放电均需检查，跳变幅度）
        if prev_row is not None and len(mono_voltage_cols) > 0:  # 存在前一行且有单体电压列
            for col in mono_voltage_cols:  # 遍历每个单体电池列
                # 获取当前行和前一行的单体电压（过滤空值）
                curr_mono = row[col] if pd.notna(row[col]) else np.nan
                prev_mono = prev_row[col] if pd.notna(prev_row[col]) else np.nan

                # 若两者均非空，计算跳变值并判断
                if not pd.isna(curr_mono) and not pd.isna(prev_mono):
                    mono_jump = abs(curr_mono - prev_mono)  # 单体电压跳变值
                    if mono_jump > MONO_VOLTAGE_JUMP_THRESHOLD:  # 超过单体电压跳变阈值
                        anomaly = True
                        reasons.append(
                            f"{col}跳变过大（上一行{prev_mono:.4f}V→本行{curr_mono:.4f}V，"
                            f"跳变{mono_jump:.4f}V > 阈值{MONO_VOLTAGE_JUMP_THRESHOLD}V）"
                        )

        # 3. 电流异常判断（仅充电时检查跳变幅度）
        if is_charge and not current_is_nan and prev_row is not None:  # 充电状态、电流非空、有前一行
            prev_current = prev_row.get('动力电池充/放电电流', np.nan)  # 前一行电流
            if not pd.isna(prev_current):  # 前一行电流非空
                current_jump = abs(current - prev_current)  # 电流跳变值
                if current_jump > CURRENT_JUMP_THRESHOLD:  # 超过电流跳变阈值
                    anomaly = True
                    reasons.append(
                        f"充电时电流跳变过大（上一行{prev_current:.2f}A→本行{current:.2f}A，"
                        f"跳变{current_jump:.2f}A > 阈值{CURRENT_JUMP_THRESHOLD}A）"
                    )

        # 4. 温度异常判断（充电/放电均需检查，跳变幅度）
        if not temp_is_nan and prev_row is not None:  # 温度非空且有前一行
            prev_temp = prev_row.get('1号温度检测点温度', np.nan)  # 前一行温度
            if not pd.isna(prev_temp):  # 前一行温度非空
                temp_jump = abs(current_temp - prev_temp)  # 温度跳变值
                if temp_jump > TEMPERATURE_JUMP_THRESHOLD:  # 超过温度跳变阈值
                    anomaly = True
                    reasons.append(
                        f"温度跳变过大（上一行{prev_temp:.2f}℃→本行{current_temp:.2f}℃，"
                        f"跳变{temp_jump:.2f}℃ > 阈值{TEMPERATURE_JUMP_THRESHOLD}℃）"
                    )

        # 记录当前行的异常结果
        anomalies.append({
            'is_anomaly': anomaly,  # 是否异常（布尔值）
            'reasons': '; '.join(reasons) if reasons else '无异常'  # 异常原因（多个原因用分号分隔）
        })

        # 更新上一行数据（用于下一行的相邻对比）
        prev_row = row

    # 统计异常总数并打印
    total_anomalies = sum(1 for a in anomalies if a['is_anomaly'])
    logger.info(f"异常检测完成，共发现 {total_anomalies} 条异常数据，异常率: {total_anomalies / len(df) * 100:.2f}%")

    return anomalies


def excel_to_json(excel_path, output_json_path, model_name, model_author):
    """
    将Excel格式的电池数据转换为JSON格式（适配大模型训练的指令格式），并添加异常标记

    参数：
        excel_path: str，输入Excel文件路径
        output_json_path: str，输出JSON文件路径
        model_name: 模型名称（用于JSON元数据）
        model_author: 模型作者（用于JSON元数据）
    返回：
        成功则返回output_json_path，失败则返回None
    """
    try:
        # 确定Excel读取引擎（.xlsx用openpyxl，.xls用xlrd，其他格式不支持）
        file_extension = os.path.splitext(excel_path)[1].lower()
        engine = 'openpyxl' if file_extension == '.xlsx' else 'xlrd' if file_extension == '.xls' else None
        if not engine:
            logger.error(f"不支持的文件格式: {file_extension}（仅支持.xlsx和.xls）")
            return None

        # 读取Excel数据
        logger.info(f"开始读取Excel文件: {excel_path}")
        df = pd.read_excel(excel_path, engine=engine)

        # 检查数据是否为空
        if df.empty:
            logger.warning(f"Excel文件为空: {excel_path}")
            return None

        # 打印数据基本信息（用于验证数据是否正确读取）
        logger.info(f"Excel数据基本信息: 行数={len(df)}, 列数={len(df.columns)}")
        logger.info(
            f"核心列检查: 状态列={'整车State状态（状态机编码）' in df.columns}, "
            f"总电压列={'动力电池内部总电压V1' in df.columns}, "
            f"单体电压列数量={sum('号电池单体电压' in col for col in df.columns)}"
        )

        # 调用异常检测函数，获取每行的异常结果
        anomalies = detect_anomalies(df)
        # 将异常结果添加到DataFrame中（便于后续保存和查看）
        df['is_anomaly'] = [a['is_anomaly'] for a in anomalies]
        df['anomaly_reasons'] = [a['reasons'] for a in anomalies]

        # 保存含异常标记的调试数据（方便人工核对异常判断是否正确）
        debug_path = os.path.splitext(output_json_path)[0] + '_debug.xlsx'
        df.to_excel(debug_path, index=False)
        logger.info(f"调试数据已保存至: {debug_path}（含异常标记列）")

        # 定义任务描述（用于构造大模型的输入指令，说明异常判断规则）
        task_description = (
            "请分析以下电池测试数据，判断是否存在异常（异常标准："
            "1.充电时：总电压跳变>3V、单体电压跳变>0.02V、电流跳变>0.5A；"
            "2.放电时：总电压>378.2V、单体电压跳变>0.02V；"
            "3.充电/放电共通：温度跳变>3℃）。"
        )

        # 转换数据为大模型训练用的JSON格式
        data = []
        for index, row in df.iterrows():
            # 拼接当前行的所有列信息（排除异常标记列，避免模型学习冗余信息）
            column_info = []
            for col in df.columns:
                if col not in ['is_anomaly', 'anomaly_reasons'] and pd.notna(row[col]):
                    column_info.append(f"{col}：{row[col]}")  # 格式："列名：值"

            if not column_info:
                continue  # 跳过空行（所有列均为NaN）

            # 构建模型输入（instruction）和输出（output）
            # 输入：任务描述 + 电池参数信息
            instruction = f"{task_description}\n" + "\n".join(column_info)
            # 输出：异常判断结果 + 原因（用于模型学习正确的判断逻辑）
            output = f"{'异常' if row['is_anomaly'] else '正常'}。{row['anomaly_reasons']}"

            # 构建样本字典（包含训练所需的所有信息）
            sample = {
                "instruction": instruction,  # 模型输入指令
                "output": output,  # 模型预期输出
                "model_name": model_name,  # 模型名称（元数据）
                "model_author": model_author,  # 模型作者（元数据）
                "is_anomaly": bool(row['is_anomaly']),  # 是否异常（用于后续统计）
                "anomaly_reasons": row['anomaly_reasons']  # 异常原因（用于后续分析）
            }
            # 添加原始列数据（可选，用于调试时回溯原始数据）
            for col in df.columns:
                # 不要覆盖前面已经写入的异常标签和异常原因
                if col in ['is_anomaly', 'anomaly_reasons']:
                    continue
                if pd.notna(row[col]):
                    sample[col] = str(row[col])  # 转为字符串避免JSON序列化问题

            data.append(sample)  # 将样本添加到数据列表

        # 保存JSON结果
        if data:
            # 用pandas将列表转为DataFrame，再保存为JSON（orient='records'表示按行存储）
            pd.DataFrame(data).to_json(output_json_path, orient='records', force_ascii=False, indent=2)
            # 统计异常样本数量
            anomaly_count = sum(1 for s in data if s['is_anomaly'])
            logger.info(
                f"数据转换完成，共{len(data)}条数据（含{anomaly_count}条异常），"
                f"已保存至: {output_json_path}"
            )

            # 单独保存异常数据报告（方便快速查看所有异常样本）
            anomaly_data = [d for d in data if d['is_anomaly']]
            if anomaly_data:
                anomaly_report_path = os.path.splitext(output_json_path)[0] + '_anomalies.json'
                pd.DataFrame(anomaly_data).to_json(anomaly_report_path, orient='records', force_ascii=False, indent=2)
                logger.info(f"异常数据单独报告已保存至: {anomaly_report_path}（含{len(anomaly_data)}条异常）")
            return output_json_path
        else:
            logger.warning("未提取到有效数据（可能所有行均为空值）")
            return None

    except Exception as e:
        # 捕获并打印异常（便于调试）
        logger.error(f"处理Excel出错: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈信息
        return None


def analyze_anomalies_from_excel(excel_path):
    """
    独立分析Excel文件中的电池数据，生成异常报告（仅分析，不进行模型训练）

    参数：
        excel_path: str，输入Excel文件路径
    """
    # 构造输出JSON路径（与Excel同目录，文件名加"_anomalies"后缀）
    output_dir = os.path.dirname(excel_path)
    base_name = os.path.basename(excel_path)
    output_json_path = os.path.join(output_dir, os.path.splitext(base_name)[0] + '_anomalies.json')

    logger.info(f"开始独立分析Excel文件: {excel_path}")
    # 调用excel_to_json转换数据并生成异常报告
    json_path = excel_to_json(excel_path, output_json_path, ['小黄', 'Xiao Huang'], ['魔搭', 'ModelScope'])

    if json_path:
        # 加载生成的JSON文件，统计并打印异常结果
        df = pd.read_json(json_path)
        if 'is_anomaly' in df.columns:
            anomaly_count = df['is_anomaly'].sum()
            total_count = len(df)
            logger.info(
                f"分析结果: 共{total_count}条数据，其中{anomaly_count}条异常，"
                f"异常率: {anomaly_count / total_count * 100:.2f}%"
            )

            # 打印前10条异常数据的原因（示例）
            if anomaly_count > 0:
                logger.info("前10条异常数据示例:")
                for i, row in df[df['is_anomaly']].head(10).iterrows():
                    logger.info(f"  异常原因: {row['anomaly_reasons']}")
        else:
            logger.warning("JSON数据中未找到异常标记列（is_anomaly）")
    else:
        logger.error("Excel分析失败（未生成JSON结果）")


if __name__ == "__main__":
    # 模式选择：True=仅分析Excel数据（不训练模型）；False=分析+模型微调
    RUN_ANALYSIS_ONLY = True
    if RUN_ANALYSIS_ONLY:
        # 模式1：仅分析Excel数据（替换为实际的Excel文件路径）
        excel_path = r'data/processed/record_core_total_confirmed_for_new_error.xlsx'  # 可改为input()实现控制台输入
        analyze_anomalies_from_excel(excel_path)
    else:
        # 模式2：完整流程（分析数据→转换为训练集→微调大模型）
        # 设置随机种子（保证实验可复现）
        seed_everything(42)

        # 配置模型和输出路径
        model_id_or_path = 'F:/qwen3/modelscope/hub/models/Qwen/Qwen3-0___6B'  # 预训练模型路径（如Qwen-6B）
        # 模型系统提示（定义模型的角色和任务）
        system = '你是电池测试数据异常检测专家，能根据提供的电池参数判断数据是否异常。'
        output_dir = 'output'  # 模型训练输出目录（保存微调后的模型、日志等）

        # 转换Excel数据为模型训练用的JSON格式
        excel_files = ['F:/data/data.xlsx']  # 输入Excel文件列表（可添加多个）
        json_output_dir = 'new_json_data'  # JSON训练集保存目录
        os.makedirs(json_output_dir, exist_ok=True)  # 创建目录（若不存在）

        dataset = []  # 存储所有转换后的JSON文件路径
        for excel_file in excel_files:
            base_name = os.path.basename(excel_file)
            json_file_name = os.path.splitext(base_name)[0] + '.json'
            json_path = os.path.join(json_output_dir, json_file_name)
            # 调用excel_to_json转换当前Excel文件
            json_path = excel_to_json(excel_file, json_path, ['小黄', 'Xiao Huang'], ['魔搭', 'ModelScope'])
            if json_path:
                dataset.append(json_path)  # 收集有效JSON路径

        # 若没有有效训练数据，退出程序
        if not dataset:
            logger.error("没有有效的JSON数据可用于训练（可能Excel处理失败）")
            exit(1)

        # 配置训练参数（控制微调过程的关键参数）
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,  # 输出目录
            learning_rate=1e-4,  # 学习率（大模型微调常用1e-4~1e-5）
            per_device_train_batch_size=2,  # 单设备训练批次大小（受显存限制）
            per_device_eval_batch_size=1,  # 单设备验证批次大小
            gradient_checkpointing=True,  # 启用梯度检查点（节省显存，会牺牲少量速度）
            weight_decay=0.1,  # 权重衰减（防止过拟合）
            lr_scheduler_type='cosine',  # 学习率调度器（余弦退火：前期大学习率快速收敛，后期小学习率微调）
            warmup_ratio=0.05,  # 预热比例（前5%的步数缓慢提升学习率，避免初始冲击）
            report_to=['tensorboard'],  # 日志记录到TensorBoard（可视化训练过程）
            logging_first_step=True,  # 记录第一步的日志
            save_strategy='steps',  # 按步数保存模型
            save_steps=50,  # 每50步保存一次模型
            eval_strategy='steps',  # 按步数验证模型
            eval_steps=50,  # 每50步验证一次
            gradient_accumulation_steps=16,  # 梯度累积步数（模拟16*2=32的大批次训练，提升稳定性）
            num_train_epochs=3,  # 训练总轮次
            metric_for_best_model='loss',  # 以验证集损失作为最优模型的评判标准
            save_total_limit=2,  # 最多保存2个模型（避免占用过多磁盘空间）
            logging_steps=5,  # 每5步打印一次训练日志
            dataloader_num_workers=1,  # 数据加载线程数（1表示单线程）
            data_seed=42,  # 数据划分的随机种子（保证训练/验证集划分一致）
        )

        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        logger.info(f'模型训练输出目录: {output_dir}')

        # 加载预训练模型和Tokenizer（分词器）
        model, tokenizer = get_model_tokenizer(model_id_or_path)
        # 获取模型输入模板（适配模型的指令格式，如Qwen的"<|im_start|>system...<|im_end|>"）
        template = get_template(model.model_meta.template, tokenizer, default_system=system, max_length=2048)
        template.set_mode('train')  # 切换模板为训练模式

        # 配置LoRA参数高效微调（只更新部分参数，降低显存占用）
        # 找到模型中所有线性层（作为LoRA的目标层，这些层对任务适配更关键）
        target_modules = find_all_linears(model)
        lora_config = LoraConfig(
            task_type='CAUSAL_LM',  # 任务类型：因果语言模型（大模型默认任务）
            r=8,  # LoRA秩（控制低秩矩阵的维度，越小参数越少）
            lora_alpha=32,  # LoRA缩放因子（调节更新幅度）
            target_modules=target_modules  # 要应用LoRA的目标层
        )
        # 为模型装配LoRA适配器（核心：将普通模型转换为可微调的LoRA模型）
        model = Swift.prepare_model(model, lora_config)

        # 加载并划分训练集和验证集（按9:1划分）
        train_dataset, val_dataset = load_dataset(
            dataset,  # 输入：JSON文件路径列表
            split_dataset_ratio=0.1,  # 验证集占比10%
            num_proc=1,  # 数据处理进程数
            seed=42  # 划分种子（保证结果一致）
        )
        logger.info(f'训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}')

        # 编码数据集（将文本指令转换为模型可识别的Token ID）
        if train_dataset:
            # 用模板预处理训练集：添加系统提示、拼接instruction和output等
            train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=1)
        if val_dataset:
            val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=1)

        # 启动模型微调
        if train_dataset and val_dataset:
            model.enable_input_require_grads()  # 启用输入梯度计算（LoRA需要）
            # 初始化训练器
            trainer = Seq2SeqTrainer(
                model=model,  # 待微调的模型（带LoRA适配器）
                args=training_args,  # 训练参数配置
                data_collator=template.data_collator,  # 数据拼接器（处理批次内的长度对齐等）
                train_dataset=train_dataset,  # 训练集
                eval_dataset=val_dataset,  # 验证集
                template=template  # 输入模板
            )
            trainer.train()  # 启动训练！
            logger.info("模型训练完成！模型已保存至输出目录")
        else:
            logger.error("训练集或验证集为空，无法启动训练")