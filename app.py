import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Student Performance Analytics", page_icon="📘", layout="wide")
st.title("📘 Student Performance Analytics")
st.caption("Interactive academic insights • Pattern exploration • Risk estimation")

@st.cache_data
def make_demo_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    attendance = rng.integers(45, 101, n)
    study_hours = np.round(rng.uniform(0, 6, n), 1)
    previous_score = rng.integers(25, 101, n)
    assignments = rng.integers(0, 11, n)
    sleep = np.round(rng.uniform(4, 10, n), 1)
    noise = rng.normal(0, 9, n)
    final_score = np.clip(0.32*previous_score + 0.45*study_hours*10 + 0.18*attendance + 1.3*assignments + 1.0*sleep + noise, 0, 100).round(0).astype(int)
    outcome = np.where(final_score >= 70, "On Track", np.where(final_score >= 50, "Needs Support", "At Risk"))
    return pd.DataFrame({"Attendance (%)":attendance, "Study Hours/Day":study_hours, "Previous Score":previous_score, "Assignments Completed":assignments, "Sleep (hours)":sleep, "Final Score":final_score, "Outcome":outcome})

df = make_demo_data()
with st.sidebar:
    st.header("Explore the demo")
    min_att = st.slider("Minimum attendance (%)", 45, 100, 45)
    outcome_filter = st.multiselect("Outcome groups", ["On Track", "Needs Support", "At Risk"], default=["On Track", "Needs Support", "At Risk"])
    st.caption("This app uses generated demo data, not real student records.")

view = df[(df["Attendance (%)"] >= min_att) & (df["Outcome"].isin(outcome_filter))]
c1,c2,c3,c4 = st.columns(4)
c1.metric("Students in view", f"{len(view):,}")
c2.metric("Average final score", f"{view['Final Score'].mean():.1f}" if len(view) else "—")
c3.metric("Average attendance", f"{view['Attendance (%)'].mean():.1f}%" if len(view) else "—")
c4.metric("At-risk share", f"{(view['Outcome'].eq('At Risk').mean()*100):.1f}%" if len(view) else "—")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔎 Relationships", "🧠 Model Evaluation", "🎯 Try a Prediction"])
with tab1:
    left,right=st.columns(2)
    with left:
        st.subheader("Outcome distribution")
        if len(view):
            counts=view["Outcome"].value_counts().reindex(["On Track","Needs Support","At Risk"]).fillna(0).reset_index()
            counts.columns=["Outcome","Students"]
            st.plotly_chart(px.bar(counts,x="Outcome",y="Students",color="Outcome",title="Students by outcome"),use_container_width=True)
        else: st.info("No students match the selected filters.")
    with right:
        st.subheader("Final score distribution")
        st.plotly_chart(px.histogram(view,x="Final Score",nbins=20,title="Final scores in filtered group"),use_container_width=True) if len(view) else st.info("No data to display.")
    st.subheader("Sample records")
    st.dataframe(view.head(25),use_container_width=True,hide_index=True)
with tab2:
    st.subheader("Explore factors associated with final scores")
    xvar=st.selectbox("Choose a factor",["Attendance (%)","Study Hours/Day","Previous Score","Assignments Completed","Sleep (hours)"])
    st.plotly_chart(px.scatter(view,x=xvar,y="Final Score",color="Outcome",trendline="ols",title=f"{xvar} vs final score"),use_container_width=True) if len(view)>2 else st.info("Choose filters that leave at least three records.")
    corr=view.select_dtypes(include="number").corr()
    st.plotly_chart(px.imshow(corr,text_auto=".2f",aspect="auto",title="Numeric feature correlation"),use_container_width=True)
with tab3:
    st.subheader("Random Forest outcome classifier")
    X=df[["Attendance (%)","Study Hours/Day","Previous Score","Assignments Completed","Sleep (hours)"]]
    y=df["Outcome"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model=RandomForestClassifier(n_estimators=150,max_depth=6,random_state=42,class_weight="balanced")
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    st.metric("Holdout accuracy",f"{accuracy_score(y_test,pred)*100:.1f}%")
    labels=["On Track","Needs Support","At Risk"]
    cm=confusion_matrix(y_test,pred,labels=labels)
    st.plotly_chart(px.imshow(cm,x=labels,y=labels,text_auto=True,color_continuous_scale="Blues",labels={"x":"Predicted","y":"Actual","color":"Count"},title="Confusion matrix"),use_container_width=True)
    imp=pd.DataFrame({"Feature":X.columns,"Importance":model.feature_importances_}).sort_values("Importance",ascending=True)
    st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h",title="Feature importance"),use_container_width=True)
    st.caption("Metrics are illustrative because the dataset is synthetically generated. This model should not be used to make real educational decisions.")
with tab4:
    st.subheader("Estimate an example student's outcome")
    a,b,c=st.columns(3)
    with a:
        attendance=st.slider("Attendance (%)",0,100,80)
        study=st.slider("Study hours/day",0.0,12.0,2.5,0.5)
    with b:
        previous=st.slider("Previous score",0,100,65)
        assignments=st.slider("Assignments completed",0,10,7)
    with c:
        sleep=st.slider("Sleep hours",0.0,12.0,7.0,0.5)
    if st.button("Estimate outcome",type="primary"):
        row=pd.DataFrame([[attendance,study,previous,assignments,sleep]],columns=X.columns)
        result=model.predict(row)[0]
        probs=model.predict_proba(row)[0]
        st.success(f"Demo estimate: **{result}**")
        st.plotly_chart(px.bar(pd.DataFrame({"Outcome":model.classes_,"Probability":probs}),x="Outcome",y="Probability",title="Model probabilities",range_y=[0,1]),use_container_width=True)
        st.warning("This is a learning demo using synthetic data, not a reliable judgement of any real student.")

st.divider()
st.caption("Academic activity demo • Python + Streamlit + scikit-learn + Plotly")
