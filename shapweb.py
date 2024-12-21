import streamlit as st
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    lgbm = joblib.load('./shapweb/lgbm.pkl')   # 本地测试路径，根据pkl文件所在位置进行更改
    # lgbm = joblib.load('./lgbm.pkl')  # 上传到github所需路径，路径无需更改

    # 定义特征输入类
    class Subject:
        def __init__(self, 认知障碍, 体育锻炼运动量, 慢性疼痛, 营养状态, HbA1c, 查尔斯共病指数, 步速下降, 糖尿病肾病):
            self.认知障碍 = 认知障碍
            self.体育锻炼运动量 = 体育锻炼运动量
            self.慢性疼痛 = 慢性疼痛
            self.营养状态 = 营养状态
            self.HbA1c = HbA1c
            self.查尔斯共病指数 = 查尔斯共病指数
            self.步速下降 = 步速下降
            self.糖尿病肾病 = 糖尿病肾病

        def make_predict(self):
            # Check for missing values (for example)
            if not all([self.认知障碍, self.体育锻炼运动量, self.慢性疼痛, self.营养状态, self.HbA1c, self.查尔斯共病指数, self.步速下降, self.糖尿病肾病]):
                st.error("Please fill in all the fields")
                return
            
            # 将输入数据转化为 DataFrame
            subject_data = {
                "认知障碍": [self.认知障碍],
                "体育锻炼运动量": [self.体育锻炼运动量],
                "慢性疼痛": [self.慢性疼痛],
                "营养状态": [self.营养状态],
                "HbA1c": [self.HbA1c],
                "查尔斯共病指数": [self.查尔斯共病指数],
                "步速下降": [self.步速下降],
                "糖尿病肾病": [self.糖尿病肾病]
            }

            df_subject = pd.DataFrame(subject_data)

            # 模型预测
            prediction = lgbm.predict_proba(df_subject)[:, 1]
            adjusted_prediction = np.round(prediction * 100, 2)
            st.write(f"""
                <div class='all'>
                    <p style='text-align: center; font-size: 20px;'>
                        <b>模型预测老年糖尿病患者衰弱风险为 {adjusted_prediction[0]} %</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # SHAP 可视化
            explainer = shap.Explainer(lgbm)
            shap_values = explainer.shap_values(df_subject)

            # 绘制 SHAP 力图
            shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], df_subject.iloc[0, :], matplotlib=True)
            st.pyplot(plt.gcf())

    # 页面配置和UI
    st.set_page_config(page_title='老年糖尿病患者衰弱风险预测')

    st.markdown(f"""
                <div class='all'>
                    <h1 style='text-align: center;'>老年糖尿病患者衰弱风险预测</h1>
                </div>
                """, unsafe_allow_html=True)

    认知障碍 = st.selectbox("认知障碍 (是 = 1, 否 = 0)", [1, 0], index=1)
    体育锻炼运动量 = st.selectbox("体育锻炼运动量 (低运动量 = 1, 中运动量 = 2, 高运动量 = 3)", [1, 2, 3], index=0)
    慢性疼痛 = st.selectbox("慢性疼痛 (有 = 1, 无 = 0)", [1, 0], index=1)
    营养状态 = st.selectbox("营养状态 (营养良好 = 0, 营养不良风险 = 1, 营养不良风险 = 2)", [0, 1, 2], index=1)
    HbA1c = st.number_input("HbA1c (mmol/L)", value=7.0, min_value=4.0, max_value=30.0)
    查尔斯共病指数 = st.number_input("查尔斯指数", value=2, min_value=0, max_value=30)
    步速下降 = st.selectbox("步速下降 (是 = 1, 否 = 0)", [1, 0], index=1)
    糖尿病肾病 = st.selectbox("糖尿病肾病 (有 = 1, 无 = 0)", [1, 0], index=1)

    if st.button(label="提交"):
        user = Subject(认知障碍, 体育锻炼运动量, 慢性疼痛, 营养状态, HbA1c, 查尔斯共病指数, 步速下降, 糖尿病肾病)
        user.make_predict()

main()
