import pandas as pd
import streamlit as st


# st.title('Calculators')
# st.header('Short Profit Calculator')
short = st.number_input(label="Short premium (i.e. $1.23)")
long = st.number_input(label="Long premium (i.e. $0.25)")
net_premium = short - long
max_profit = st.number_input(label="Max Profit (i.e. $98.0)", value=net_premium * 100)
qty = st.number_input(label="Number of Contracts", value=1)

profit_margin = [100, 75, 50, 25, 0]
highlight = 50

df = pd.DataFrame({"Profit": profit_margin})
df["Margin"] = max_profit / qty / 100 * (1 - df["Profit"] / 100)
color = "background-color: #3d3d3d;"
df = (
    df.style.hide(axis="index")
    .format(precision=0, formatter={"Profit": "{:.0f}%", "Margin": "${:.2f}"})
    .apply(lambda i: [color if j == highlight else "" for j in i])
)
st.table(df)
