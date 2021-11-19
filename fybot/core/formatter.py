"""Standards for df styles
df = (df.style.background_gradient(axis=0, subset=list(gradient_cols))
              .highlight_max(subset=list(highlight_cols),color='darkgreen')
              .highlight_min(subset=list(highlight_cols),color='darkred')
              .format({
                        'Return':  "{:.2%}",
                        'Daily Return': "{:.2%}",
                        'Max Profit': "${:,.2f}",
                        'Risk': "${:,.2f}",
                        'Qty': "{:0.0g}x",
                        'Mark': "${:.2f}",
                        'Delta': "{:.2f}",
                        'IV': "{:.0f}"
                        })
      )
"""
HI_MAX_COLOR = 'darkgreen'
HI_MIN_COLOR = 'darkred'
FMT_PERCENT0 = "{:.0%}"
FMT_PERCENT2 = "{:.2%}"
FMT_FLOAT = "{:,.2f}"
FMT_DOLLAR = "${:,.2f}"
FMT_INTEGER = "{:0.0g}"
