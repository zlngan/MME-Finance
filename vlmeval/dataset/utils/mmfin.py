from vlmeval.smp import *
from vlmeval.utils import track_progress_rich
import re
import time

def build_mmfin_gpt4_prompt(line):
    question = line['question']
    gt = str(line['answer'])
    prediction = str(line['prediction'])

#add id
    prompt = """
You are a financial expert who is well-versed in various financial charts and has extensive financial knowledge. 
Now you are given a chart picture, corresponding question, ground truth and prediction.
Compare the ground truth and prediction from AI models, to give a correctness score for the prediction.
The final answer must be an integer score. If a decimal is obtained, it can be rounded.
"""

# add requirement
    if line['task_category'] == 'OCR' or line['task_category'] == 'Entity Recognition' or line['task_category'] == 'Spatial Awareness':
        prompt += """
This is an objective question. Please give a score: (The full score is 5 points in total. Score according to the following conditions.)
Answer accuracy: Full score is 5 points. In combination with the question, it is required that the content and semantics of the prediction and the answer must be the same and there should be no redundant answers. The answer can be expressed in different ways, such as different unit symbols and different counting methods. If the answer is correct, 5 points can be given. If the answer contains multiple pieces of content, multiply 5 by the correct proportion of the prediction to give the final score. If the answer is wrong, give 0 points directly.
"""
    if line['task_category'] == 'Accurate Numerical Calculation' :
        prompt += """
This is a calculation question. Please give a score: (The full score is 5 points in total. Score item by item according to the following conditions and add up the obtained scores to get the total score.)
1. Answer accuracy: Full score is 2 points. In combination with the question and answer, it is required that the final calculated result of the prediction must be accurate. If the answer is correct, give 2 points. If the answer is wrong, give 0 points.
2. Calculation process: Full score is 3 points. There should be intermediate calculation processes for calculation questions, and they should also be correct. In combination with the answer, if all elements and steps are included in the prediction, give 3 points. If the final answer is wrong but the calculation process included in the prediction is partially correct, multiply 3 by the correct proportion to give the final score. If the calculation process is also wrong, give 0 points.
"""
    if line['task_category'] == 'Numerical Calculation' :
        prompt += """
This is a valuation calculation question. Please give a score: (The full score is 5 points in total. Score item by item according to the following conditions and add up the obtained scores to get the total score.)
1. Answer accuracy: Full score is 2 points. In combination with the question and answer, if the predicted final result fluctuates within ±10% of the final value of the answer, give 2 points. If the predicted final result fluctuates within ±10% - ±20% of the final value of the answer, give 1 point. If the predicted final result fluctuates more than ±20% of the final value of the answer, give 0 points.
2. Calculation process: Full score is 3 points. There should be intermediate calculation processes for calculation questions, and they should also be correct. In combination with the answer, if all elements and steps are included in the prediction, give 3 points. If the final answer is wrong but the calculation process included in the prediction is partially correct, multiply 3 by the correct proportion to give the final score. If the calculation process is also wrong, give 0 points.
"""
    if line['task_category'] == 'Image Caption' or line['task_category'] == 'Financial Knowledge' or line['task_category'] == 'Risk Warning' or line['task_category'] == 'Investment Advice' or line['task_category'] == 'Explain Reason': 
        prompt += """
This is a subjective question. Please give a score: (The full score is 5 points in total. Score item by item according to the following conditions and add up the obtained scores to output the total score.)
1. Content matching degree: Full score is 2 points. When all keywords of the answer appear in the predicted text, give 2 points. When some keywords of the answer appear in the predicted text, give 1 point. When none of the keywords of the answer appear in the predicted text, give 0 points.
2. Semantic matching degree: Full score is 2 points. When the semantics of the answer and the predicted content are close and there is no wrong judgment, give 2 points. When part of the semantics of the answer and the predicted content are close, give 1 point. When the semantics of the answer and the predicted content are completely different, give 0 points.
3. Problem attribute self-consistency: Full score is 1 point. Give points as appropriate according to the following prediction requirements, and require smooth logic and correct grammar.
"""
    if line['task_category'] == 'Not Applicable': 
        prompt += """
This is an unanswerable question. Please give a score: (The full score is 5 points in total. Score according to the following conditions.)
Answer accuracy: Full score is 5 points. If 'Not Applicable' appears in the prediction result, give 5 points directly. If 'Not Applicable' does not appear in the prediction but indicates that it cannot be answered, give points as appropriate.
"""
    prompt += 'Here are the prediction requirements: '
    format_prompt = {
        'Image Caption':'Describe this image in whole-part structure. Start with a sentence summarizing the main theme of the image. If the image describes multiple objects, firstly introduce each object in one sentence, and if there are some connections between objects, explain every connection in one sentence. If the object is complex, it can be further explained. Your answer should be less than 250 words and should not include any irrelevant information.\n',
        'OCR':'The answer should conduct an Optical Character Recognition (OCR) analysis on the content asked about. Just answer the question with a single word or phrase if possible. No irrelevant information should be included.\n',
        'Entity Recognition':'The answer should  contain recognition results of entities mentioned in the question. Just answer the question with a single word or phrase if possible. No irrelevant information should be included.\n',
        'Spatial Awareness':"The answer should be based on the spatial relationships between entities in the qusition. It is best to provide corresponding evidence for all judgments. If specific numerical answers are not present in the image but can be estimated based on its content, the estimated results can be used. Just provide the answer in one word or a short sentence. No irrelevant information should be included.\n",
        'Numerical Calculation':'You should perform mathematical calculations based on the information in the image. You need to estimate some values that do not display directly in the image for answering the question. You should show the calculation process and output the calculated result.\n',
        'Accurate Numerical Calculation':'You should perform mathematical calculations based on the information in the image. You need to provide a step-by-step calculation and obtain a numerical result.\n',
        'Financial Knowledge':'The answer should be based on financial knowledge. Briefly answer the question within100 words. The answer should not contain irrelevant content related to the picture.\n',
        'Risk Warning':'You should warn investment risk based on the information in the chart and professional financial knowledge. All your arguments need to be supported by facts or theories and the answer should fall within 150 words.\n',
        'Investment Advice':'You should provide investment advice based on the information in the chart and professional financial knowledge. All your arguments need to be supported by facts or theories and the answer should fall within 150 words.\n',
        'Explain Reason':'You should provide explanation based on the information in the chart and professional financial knowledge. All your arguments need to be supported by facts or theories and the answer should fall within 150 words.\n',
        'Not Applicable':'If you cannot answer, please say "Not Applicable", and provide the explainations.',
    }
    prompt += format_prompt[line['task_category']] if line['task_category'] in format_prompt else ''

# add example
    prompt += """
The correctness score is 0(totally wrong), 1, 2, 3, 4, 5(totally right).
Just complete the last space of the correctness score.
Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
"""
    example_prompt1 = {
        'Image Caption':'What does the image depict? | The image depicts a technical analysis chart of an asset using the KDJ indicator. The KDJ indicator is a momentum oscillator that includes three lines: - K Line (Yellow): Measures the relative position of the closing price within a certain period. - D Line (Cyan): Typically a moving average of the K Line. - J Line (Pink): Represents the divergence between the D Line and K Line. The values of K, D, and J shown in the chart are 57.06, 63.47, and 44.22, respectively. Horizontal Axis: Time scale ranging from November of the previous year to mid-June of the current year. | The main subject of the picture is a display of financial data. The picture shows various technical indicators such as KJ(9, 3, 3) with specific values for K, D, and J. There are also indicators like Indicator de and a value of +49.2. Additionally, a setting section mentions multiple financial indicators like MACD, KDJ, RSI, BOLL, W&R, DMI, BIAS, ASI, VR, ARBR, DPO, TRIX, New DMA, BBI, MTM, OBV, SAR, and EXPMA with specific values. The connection between these elements is that they are all part of a comprehensive set of financial indicators used to analyze and understand the financial market. The values associated with these indicators convey specific information about market trends, momentum, volatility, and other characteristics that can help traders and investors make informed decisions. | 3 ',
        'OCR':'What is the operating cash flow value in 2024 Q4? | 11.50 billion | 11.50B | 5 ',
        'Entity Recognition':'What sections are included in this document? | 1. Earnings per Share 2. Stock-based Compensation 3. Comprehensive Income 4. Derivatives and Hedging Activities 5. Credit Risk | Net income attributable to 3M, Denominator, Earnings per share attributable to 3M common shareholders—basic, Earnings per share attributable to 3M common shareholders—diluted, Stock-based compensation, Comprehensive income, Derivatives and hedging activities, Credit risk. | 4 ',
        'Spatial Awareness':'Which quarter shows the highest institutional holding ratio? | 2024 Q2 / 2023 Q1 | 2024 Q2 | 5 ',
        'Numerical Calculation':'By how much did the Institutional Holding Ratio change from 2022 Q1 to 2022 Q3? | The Institutional Holding Ratio for OPTT was approximately 14% in Q1 2022. In Q3 2022, the Institutional Holding Ratio had decreased to about 8%. Hence the change is 14% - 8% = 6%. | To calculate the change in the Institutional Holding Ratio from 2022 Q1 to 2022 Q3, we subtract the ratio for 2022 Q3 from the ratio for 2022 Q1. The ratio for 2022 Q1 is approximately 12.19%, and for 2022 Q3, it is approximately 16.00%. The calculation is as follows: 16.00% - 12.19% = 3.81%./nThe Institutional Holding Ratio increased by approximately 3.81% from 2022 Q1 to 2022 Q3. | 1 ',
        'Accurate Numerical Calculation':'How much do the stock prices of Microsoft and Google differ? | The share price of GOOGL was 184.49, the share price of MSFT was 456.73. So for the difference of these two stock prices was 456.73 - 184.49 = 272.24. | To find the difference in stock prices between Microsoft (MSFT) and Google (GOOGL), we subtract the closing price of Google from the closing price of Microsoft. The closing price of MSFT is 456.73 and the closing price of GOOGL is 184.49. The difference is 456.73 - 184.49 = 272.24./nThe difference in stock prices between Microsoft and Google is 272.24. | 5 ',
        'Financial Knowledge':'Explain what operating cash flow is. | Operating cash flow (OCF) is how much cash a company generated (or consumed) from its operating activities during a period. | Operating cash flow refers to the cash inflows and outflows generated by a company\'s core business operations. It is a key indicator of a company\'s financial health and liquidity, as it reflects the ability to generate cash from day-to-day business activities. Operating cash flow is calculated by subtracting operating expenses, including depreciation and amortization, from operating income. A positive operating cash flow suggests that a company is generating sufficient cash to cover its expenses and invest in growth, while a negative operating cash flow indicates a cash shortfall that may require external financing or cost-cutting measures./nOperating cash flow is a measure of a company\'s ability to generate cash from its core business operations, calculated by subtracting operating expenses from operating income. | 2 ',
        'Risk Warning':'Based on the information in the RSI chart, what potential risks might exist? | The potential risks mentioned in the analysis hint are near overbought condition, underbought condition, crossing lines, consistency, historical context, and volatility. These risks suggest that the market is currently experiencing high volatility and potential changes in trend. It is recommended to monitor the RSI closely for any signs of reversal in the market trend.| Not Applicable. The provided image does not contain an RSI (Relative Strength Index) chart. Therefore, it is not possible to analyze the potential risks based on the information in the chart. The RSI is a momentum oscillator that measures the speed and change of price movements to determine overbought or oversold conditions. Without the RSI chart, any analysis of investment risks would be speculative and not based on the evidence presented in the image./nSince the RSI chart is not present in the provided image, it is not possible to analyze the potential risks based on the information in the chart. | 0 ',
        'Investment Advice':'Based on the information in the table, what investment advice would you give? | Based on the information presented in the table and chart, here is a detailed analysis and investment advice for Intel Corporation (INTC): Stock Overview: - Current Stock Price: 20.130 - Change: +0.520 (+2.65%) - Volume: 48.257K. Candlestick Chart: - Opening Price: 19.76 - Highest Price: 20.60 - Lowest Price: 19.75 - Closing Price: 20.130 (Increase by 0.52 or +2.65%). Technical Indicators: - Moving Averages: - MA5: 30.826 - MA20: 32.708 - MA60: 31.284 - MA120: 35.653 - MA250: 37.596 - RSI: - RSI6: 20.72 - RSI12: 33.69 - RSI24: 40.79. Financial Metrics: - Market Value: 86.08 Billion - PE (TTM): 84.50 - Dividend: 0.13 (yield 2.48%) - Beta: 1.05. Observations: 1. Trend: The stock has experienced a significant downtrend, evident by the recent lows and high trading volume spikes. 2. Volume: The increase in trading volume could indicate rising investor interest or concern, typically observed when nearing support or resistance levels. 3. RSI Levels: The RSI values are relatively low (below 30 for some intervals), indicating that the stock might be oversold, suggesting a potential buying opportunity. 4. Moving Averages: The stock price is below short, mid, and long-term moving averages, indicating bearish sentiment. Investment Advice: Given the current scenario: 1. Risk Analysis: The stock is in a downtrend with significant bearish sentiment. However, the increase in volume and the low RSI indicate it might be hitting a support level and could be poised for a bounce. 2. Short-Term Strategy: If you are a short-term trader, you may consider entering a position to capitalize on a potential bounce from oversold conditions. Ensure to set stop-loss orders to manage risks in case of further declines. | To provide investment advice based on the chart, one would need to analyze various indicators such as moving averages, relative strength index (RSI), and volume. For example, if the RSI is below 30, it might indicate oversold conditions, suggesting a potential buying opportunity. Conversely, if the RSI is above 70, it could indicate overbought conditions, suggesting a potential selling opportunity. Additionally, if the volume is increasing while the price is falling, it could be a sign of increased selling pressure, which might indicate a bearish trend./nInvestment advice would depend on the specific indicators and trends observed in the chart. For instance, if the chart shows a bullish trend with increasing volume, it might be advisable to consider buying the stock. However, if the chart shows a bearish trend with decreasing volume, it might be advisable to consider selling the stock. It is important to note that these are hypothetical examples and actual investment decisions should be based on a thorough analysis and possibly consultation with a financial advisor. | 1 ',
        'Explain Reason':'What potential risks can be inferred from the RSI in the table? | The table shows several stocks with an RSI (Relative Strength Index) of 100 as of August 16, 2024. An RSI value of 100 indicates that the stocks are in an extremely overbought condition. This can pose several potential risks: 1. Aogo Capital-U (AOGOU) - Last Price: 10.97 - RSI: 100.00 - Risk: Highly overbought, with a potential for price correction or reversal. 2. Flag Ship (FSHP) - Last Price: 10.00 - RSI: 100.00 - Risk: Highly overbought, with a potential for price correction or reversal. 3. M3-Brigade-U (MBAV.U) - Last Price: 10.05 - RSI: 100.00 - Risk: Highly overbought, with a potential for price correction or reversal. 4. ExcelFin-U (XFINU) - Last Price: 11.00 - RSI: 100.00 - Risk: Highly overbought, with a potential for price correction or reversal. 5. Four Leaf-U (FORLU) - Last Price: 11.03 - RSI: 100.00 - Risk: Highly overbought, with a potential for price correction or reversal. 6. Goldenstone-U (GDSTU) - Last Price: 11.30 - RSI: 100.00 - Risk: Highly overbought, with a potential for price correction or reversal. 7. Global Technology-U (GTACU) - Last Price: 10.99 - RSI: 100.00 - Risk: Highly overbought, with a potential for price correction or reversal. In summary, all stocks listed in the table have an RSI of 100, indicating they are in a highly overbought condition. This suggests a high likelihood of these stocks experiencing a decline or correction in the near future as investors may sell off shares to realize profits. Investors should consider this risk and look for signs of a trend reversal. | With all stocks in the table having an RSI of 100.00, a potential risk is that the market may be overbought. An RSI of 100 typically indicates that the asset has been in strong upward momentum, but this could lead to a reversal. For example, in the case of AOGOU Arogo Capital-U, there are risks related to compliance with Nasdaq\'s market value requirements. This shows that even if the stock appears strong in the short term based on a high RSI, there are underlying structural risks that could impact its future performance. | 3 ',
        'Not Applicable':'What does the yellow curve represent in the RSI chart? | There is no yellow curve in the RSI chart, hence the answer is not applicable. | Since there is no yellow curve in the RSI chart provided, it is not possible to determine what it represents. However, if there were a yellow curve, it could potentially represent a specific indicator within the RSI framework, such as a moving average or a particular period\'s RSI value. For example, it might be the RSI12 indicator, which is the 12-period Relative Strength Index used to measure the speed and change of price movements and help identify overbought or oversold conditions in the market./nIf there were a yellow curve in the RSI chart, it could potentially represent an important indicator within the RSI framework. | 0 ',
    }
    prompt += example_prompt1[line['task_category']] if line['task_category'] in example_prompt1 else ''
    example_prompt2 = {
        'Image Caption':'What does the image represent? | The image represents a stock market table detailing comprehensive information about various companies. The columns provide key metrics such as: 1. No.: The rank or serial number of the company in the list. 2. Ticker: The stock ticker symbol, an abbreviation used to uniquely identify publicly traded shares of a company. 3. Company Name: The full name of the company. 4. Price: The current stock price of the company. 5. Chg. (%): The percentage change in the stock price. 6. Market Cap (USD): The market capitalization of the company in U.S. dollars as of July 1, 2024, indicating the total market value of a company\'s outstanding shares. 7. Close (USD): The closing price in U.S. dollars as of July 1, 2024. 8. TO.rate (%): The turnover rate percentage as of July 1, 2024, indicating the trading volume of shares in relation to the number of shares outstanding. 9. Vol (Share): The volume of shares traded on July 1, 2024. 10. **$VOL (USD)**: The dollar volume of shares traded on July 1, 2024. 11. GICS Industry: The Global Industry Classification Standard (GICS) sector to which the company belongs. 12. Open (USD): The opening price of the stock on July 1, 2024. | The main subject of the picture is a table of financial data for different companies. Each row represents a company with details such as ticker symbol, company name, price, percentage change, market cap, various financial ratios, industry, and opening price. The connection between these objects is that they are all part of a comprehensive view of multiple companies\' financial positions on a specific date. The elements like price and percentage change convey the current value and movement of a company\'s stock. Market cap shows the company\'s size in the market. Industry classification helps in understanding the sector in which the company operates. These elements together provide valuable information for investors to analyze and make decisions about these companies. | 4 ',
        'OCR':'What are the last candlestick values for SMA 5/10/20/30/60? | The last candlestick values for the Simple Moving Averages (SMA) of periods 5, 10, 20, 30, and 60 for Microsoft Corp. (MSFT) are as follows: - SMA 5: 448.09 - SMA 10: 444.67 - SMA 20: 433.29 - SMA 30: 430.03 - SMA 60: 420.50 | 5:448.09, 10:444.67, 30:430.03, 60:420.50 | 4 ',
        'Entity Recognition':'What does the blue bar represent in the chart? | Institutional Holding Ratio | The blue bar represents the Institutional Holding Ratio./nThe blue bar represents the Institutional Holding Ratio. | 5 ',
        'Spatial Awareness':'Analyze the tendency of the K curve from June 1st. | The K curve first rose to close to 90 in early June, then fell back to less than 50, and finally rose again. | The K curve shows a fluctuating trend from June 1st, initially rising, then falling, and finally rising again./nThe K curve shows a fluctuating trend from June 1st. | 2 ',
        'Numerical Calculation':'Analyze the trend of total impairment losses recognized from 2021 to 2023. | From 2021 to 2022, total impairment is on the rise, and from 2022 to 2023, it is on the decline. | The total impairment losses recognized in the consolidated statements of income for the years ended June 30, 2023, 2022, and 2021 were $18 million, $43 million, and $10 million, respectively. The trend shows an increase from 2021 to 2022, followed by a decrease in 2023./nThe trend shows an increase from 2021 to 2022, followed by a decrease in 2023. | 5 ',
        'Accurate Numerical Calculation':'How much do the stock prices of Microsoft and Google differ? | The share price of GOOGL was 184.49, the share price of MSFT was 456.73. So for the difference of these two stock prices was 456.73 - 184.49 = 272.24. | To find the difference in stock prices between Microsoft (MSFT) and Google (GOOGL), we subtract the closing price of GOOGL from the closing price of MSFT. The closing price of MSFT is $456.73, and the closing price of GOOGL is $182.99. The difference is $456.73 - $182.99 = $273.74./nThe difference in stock prices between Microsoft and Google is $273.74. | 2 ',
        'Financial Knowledge':'What information does the balance sheet provide about the company? | A balance sheet provides a snapshot of a company\'s financial condition at a specific point in time. The provided balance sheet indicates several key metrics and trends: 1. Debt Ratio: The debt ratio is charted over five quarters. The debt ratio measures the proportion of the company\'s assets that are financed by debt. It starts at 44.85% in Q1 2024, showing a slight decrease over the subsequent quarters, and reaches 36.24% in Q1 2025. 2. Assets: Represented by the blue bars, total assets fluctuate over the reported quarters, showing an increase from Q1 2024 to Q2 2024, a dip in Q3 2024, followed by another increase in Q4 2024, and peaking again in Q1 2025. 3. Liabilities: Illustrated by the light blue bars, liabilities are fairly consistent over the time period, with a slightly downward trend. 4. Overall Financial Health: The decreasing debt ratio in conjunction with relatively stable liabilities and varying assets can be interpreted as an indication of improving financial health. | The balance sheet provides information on the company\'s debt ratio over different quarters. It shows the relationship between assets, liabilities, and equity. For example, in 2024 Q2, the debt ratio is 56% with specific asset and liability values. This helps assess the company\'s financial stability and leverage. | 1 ',
        'Risk Warning':'Based on the information in the image, what potential risks could be identified? | The insider transactions report indicates multiple significant stock sales by the President and CEO, HUANG JEN HSUN, on the same day, June 27, 2024. This pattern raises potential red flags for investors and stakeholders. Such extensive stock offloading by a top executive can signal a lack of confidence in the company\'s future performance, possible anticipation of bad news or a perception that the company\'s current valuation is relatively high. This could lead to volatility and a potential decline in stock prices as investors may react negatively, interpreting the sales as an indication of internal issues or forthcoming negative developments. Coupled with the company\'s dependency on a narrow tech market segment and potential political uncertainties highlighted in the background information, these insider sales contribute to increased risks for the company\'s stock performance. Investor sentiment could be further impacted by concerns over CEO stability and corporate strategy continuity amid these transactions. | Based on the information in the chart, potential risks that could be identified include the impact of CEO changes on stock price, as well as the potential for insider trading. The chart shows that there have been multiple insider transactions within a short period of time, which could raise concerns about potential conflicts of interest or insider trading. Additionally, the changes in CEO could potentially impact the company\'s performance and stock price, as investors may have different expectations for the new leadership. It is important for investors to carefully consider these risks and conduct thorough research before making investment decisions./nBased on the information in the chart, potential risks that could be identified include the impact of CEO changes on stock price, as well as the potential for insider trading. The chart shows that there have been multiple insider transactions within a short period of time, which could raise concerns about potential conflicts of interest or insider trading. Additionally, the changes in CEO could potentially impact the company\'s performance and stock price, as investors may have different expectations for the new leadership. It is important for investors to carefully consider these risks and conduct thorough research before making investment decisions. | 0 ',
        'Investment Advice':'Based on the information in the table and background, what investment advice would you give? | Based on the information in the table, here is an investment analysis for Walmart (WMT): 1. 52 Week Range: The stock has traded between $49.85 and $76.22 over the past year, indicating a significant range of price movement. The current price of $75.60 is near the upper end of this range, suggesting the stock is trading close to its 52-week high. 2. Day Range: The stock\'s intraday price range is between $75.15 and $76.22, showing some volatility within the day. 3. Volume: With 16.51 million shares traded, there is a high level of liquidity, which is positive for investors looking to enter or exit positions. 4. Turnover: The total dollar value of shares traded is 1.25 billion, further indicating strong trading activity. 5. P/E (TTM) and P/E (Static): The Price to Earnings ratios are 39.09 and 39.19, respectively. These are relatively high, suggesting that the stock may be overvalued compared to the market average. Investors should consider whether the company\'s growth prospects justify this valuation. 6. Turnover %: At 0.38%, this indicates a relatively low turnover rate, implying that a small percentage of the company\'s market cap is traded daily. 7. Market Cap: Walmart\'s market capitalization is 607.93 billion dollars, making it a large-cap stock with a stable market presence. 8. P/B Ratio: The Price to Book ratio is 7.20, which is higher than what value investors typically look for (below 3.0). This suggests the stock might be overvalued in terms of its book value. 9. Shares Outstanding: There are 8.04 billion shares outstanding, indicating a large number of shares available in the market. ### Investment Advice: - Growth Potential: Given the high P/E ratio, investors should assess Walmart\'s growth potential. If the company can sustain or accelerate its growth, the high valuation might be justified. - Valuation Concerns: The high P/E and P/B ratios suggest that the stock might be overvalued. Investors should compare these ratios with industry peers and consider whether Walmart\'s fundamentals support such valuations. - Market Position: Walmart\'s large market cap and high trading volume indicate a strong market position and liquidity, which are positive factors for stability and ease of trading. - Risk Assessment: Given the stock\'s proximity to its 52-week high, there might be limited upside in the short term. Investors should be cautious of potential corrections. - Long-Term Investment: For long-term investors, Walmart\'s strong market presence and liquidity make it a relatively safe investment. However, they should keep an eye on the company\'s growth metrics and market conditions. In summary, while Walmart is a stable and liquid investment, potential investors should carefully consider its high valuation ratios and assess whether the company\'s growth prospects justify the current stock price. | Based on the information in the chart, Walmart\'s stock price is currently at $75.60, which is higher than its 52-week low of $49.85 and lower than its 52-week high of $76.45. The stock beta is 0.50, indicating that it is less volatile than the broader market. The company\'s P/E ratio is 39.48, which is higher than the P/E ratios of its competitors Costco, Target, and BJ Wholesale Club. However, Walmart\'s P/B ratio is 4.71, which is higher than its competitors\' P/B ratios, indicating that the stock may be overvalued relative to its book value. Overall, the chart suggests that Walmart\'s stock may be overvalued and investors should consider this when making investment decisions. | 0 ',
        'Explain Reason':'What are the main reasons for the decrease in company employees in the past two years? | The main reasons for the decrease in company employees over the past two years can be summarized as follows: 1. Global macroeconomic factors and China\'s stringent Covid Zero policy have affected consumer spending and the company\'s growth. 2. The company has been reducing headcount to improve cost efficiency and adapt to a slower economy. These reasons are supported by the following evidence: 1. The net profit has shown significant volatility over the past two years, with a steep decline to around 65,000 in 2022-03-31, followed by a recovery to around 93,000 by 2023-03-31 and a slight decrease to around 86,000 by 2024-06-30. This suggests that the company has been facing challenges in maintaining profitability. (Reference: Background) 2. Alibaba Group Holding Ltd. cut its staff by roughly 20,000 over the course of 2023, adding to a spate of layoffs spanning global tech companies from Silicon Valley to Hangzhou. The company\'s pace of eliminating roles has been fairly steady, as it reduced headcount by roughly the same number over the course of 2022 as well. (Reference: Alibaba Reduced Headcount by a Further 20,000 in 2023) 3. The company\'s growth has been stunted by global macroeconomic factors and China\'s stringent Covid Zero policy, with lockdowns blunting consumer spending. Alibaba said in May that it will take a “more disciplined” approach to spending and scale back expenses in areas that are not generating long-term value. This shift — in line with Beijing incentives — marks a major change from the aggressive and wide-ranging market grab that characterized the e-commerce giant in the past. (Reference: Alibaba laid off 19,000 employees in 2022 to improve cost...) In summary, the decrease in company employees over the past two years can be attributed to a combination of economic factors and the company\'s efforts to improve cost efficiency and adapt to a slower economy. | Based on the information in the chart, the main reasons for the decrease in company employees over the past two years can be attributed to the company\'s efforts to improve cost efficiency and adapt to a global economic chill. In 2022, Alibaba Group Holding Ltd. reduced its workforce by roughly 19,000 employees, and in 2023, the company cut its staff by roughly 20,000 over the course of the year. The company\'s growth has been stunted by global macroeconomic factors and China\'s stringent Covid Zero policy, with lockdowns blunting consumer spending. Alibaba said in May that it will take a "more disciplined" approach to spending and scale back expenses in areas that are not generating long-term value. This shift, in line with Beijing incentives, marks a major change from the aggressive and wide-ranging market grab that characterized the e-commerce giant in the past. Alibaba, which still has a workforce of 239,740, per its latest report, remains one of China\'s biggest private-sector employers. It reported sales and profit that beat average analyst expectations and its shares surged in response. Still, its job cutbacks in December increased from September, so the company seems to still be adjusting its size to deal with a slower economy — even with growing signs of a recovery taking root at home and abroad. | 4 ',
        'Not Applicable':'Analyze the sales revenue trend of Carrefour. | Based on the chart, it is impossible to answer this question. Hence, the answer is not applicable. | Not Applicable. The provided image does not contain a chart or any visual representation of the revenue over a 10-year period. Therefore, it is not possible to determine how the revenue has changed based on the information given in the image./nThe information in the image is not sufficient to answer the question about the revenue change over 10 years. | 5 ',
    }
    prompt += example_prompt2[line['task_category']] if line['task_category'] in example_prompt2 else ''
    gpt4_prompt = prompt + '\n' + ' | '.join(
        [question, gt.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> '), prediction, ''])
    return gpt4_prompt

def find_last_int_in_range(s):
    pattern = r'[-+]?\d+'
    matches = re.findall(pattern, s)
    ints = [int(m) for m in matches]
    in_range_ints = [i for i in ints if 0 <= i <= 5]
    return in_range_ints[-1] if in_range_ints else None

import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def MMfin_auxeval(model, line):    
    def int_cvt(s):
        try:
            return int(s)
        except ValueError:
            return find_last_int_in_range(s)

    prompt = build_mmfin_gpt4_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        mes = [{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': prompt,
                }, {
                    'type': 'image_url',
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(osp.join(LMUDataRoot(), 'images/MMfin', line['image_path']))}"
                    }
                }],
            }]
        output = model.generate(mes, temperature=i * 0.5)
        score = int_cvt(output.choices[0].message.content)
        if score is None:
            log += f'Try {i}: output is {output}, failed to parse.\n'
        elif score < 0 or score > 5:
            log += f'Try {i}: output is {output}, invalid score: {score}.\n'
        else:
            log += f'Try {i}: output is {output}, parse succeed'
            return dict(log=log, score=score)
    print(f'All {retry} retries failed.')
    print('log: ',log)
    log += f'All {retry} retries failed.\n'
    return dict(log=log, score=0.0)


def build_mmfin_gpt4_prompt_cn(line):
    question = line['question']
    gt = str(line['answer'])
    prediction = str(line['prediction'])

#add id
    prompt = """
你是一位精通各种金融图表且拥有广泛金融知识的金融专家。现在给你一张图表图片、相应的问题、真实答案和人工智能模型的预测结果。比较真实答案和人工智能模型的预测结果，为预测结果给出一个正确性得分。最终答案必须是一个整数得分。如果得到小数，可以进行四舍五入。
"""

# add requirement
    if line['task_category'] == 'OCR' or line['task_category'] == 'Entity Recognition' or line['task_category'] == 'Spatial Awareness':
        prompt += """
这是一道客观题。请给出一个分数：（总分为5分。根据以下条件进行评分。）
答案准确性：满分为5分。结合问题，要求预测结果和答案的内容及语义必须相同，且不能有多余的答案。答案可以用不同的方式表达，例如不同的单位符号和不同的计数方法。如果答案正确，可以给5分。如果答案包含多部分内容，则用5乘以预测正确的比例给出最终分数。如果答案错误，直接给0分。
"""
    if line['task_category'] == 'Accurate Numerical Calculation' :
        prompt += """
这是一道计算题。请给出一个分数：（总分为5分。根据以下条件逐项评分并将所得分数相加得到总分。）
1.答案准确性：满分为2分。结合问题和答案，要求预测的最终计算结果必须准确。如果答案正确，给2分。如果答案错误，给0分。
2.计算过程：满分为3分。计算题应该有中间计算过程，并且也应该是正确的。结合答案，如果预测中包含了所有的元素和步骤，给3分。如果最终答案错误但预测中包含的计算过程部分正确，用3乘以正确的比例给出最终分数。如果计算过程也错误，给0分。
"""
    if line['task_category'] == 'Numerical Calculation' :
        prompt += """
这是一道估值计算题。请给出一个分数：（总分为5分。根据以下条件逐项评分并将所得分数相加得到总分。）
1.答案准确性：满分为2分。结合问题和答案，如果预测的最终结果在答案最终值的±10%范围内波动，给2分。如果预测的最终结果在答案最终值的±10%至±20%范围内波动，给1分。如果预测的最终结果在答案最终值的±20%范围外波动，给0分。
2.计算过程：满分为3分。计算题应该有中间计算过程，并且也应该是正确的。结合答案，如果预测中包含了所有的元素和步骤，给3分。如果最终答案错误但预测中包含的计算过程部分正确，用3乘以正确的比例给出最终分数。如果计算过程也错误，给0分。
"""
    if line['task_category'] == 'Image Caption' or line['task_category'] == 'Financial Knowledge' or line['task_category'] == 'Risk Warning' or line['task_category'] == 'Investment Advice' or line['task_category'] == 'Explain Reason': 
        prompt += """
这是一道主观题。请给出一个分数：（总分为5分。根据以下条件逐项评分并将所得分数相加输出总分。）
1.内容匹配度：满分为2分。当答案的所有关键词都出现在预测文本中时，给2分。当答案的部分关键词出现在预测文本中时，给1分。当答案的关键词都没有出现在预测文本中时，给0分。
2.语义匹配度：满分为2分。当答案和预测内容的语义相近且没有错误判断时，给2分。当答案和预测内容的部分语义相近时，给1分。当答案和预测内容的语义完全不同时，给0分。
3.问题属性自洽性：满分为1分。根据以下预测要求适当给分，要求逻辑通顺、语法正确。
"""
    if line['task_category'] == 'Not Applicable': 
        prompt += """
这是一道无法回答的问题。请给出一个分数：（总分为5分。根据以下条件进行评分。）
答案准确性：满分为5分。如果预测结果中出现 “不适用”，直接给5分。如果预测中未出现 “不适用” 但表明无法回答问题，可适当给分。
"""
    prompt += '这是答案的一些要求：'
    format_prompt = {
        'Image Caption':'以整体-部分结构描述此图像。先以一句话概括图像的主要主题。如果图像描述多个对象，首先用一句话介绍每个对象，如果对象之间存在一些联系，用一句话解释每个联系。如果对象很复杂，可以进一步解释。你的回答应少于 250 个字，且不应包含任何无关信息。\n',
        'OCR':'答案应对所询问的内容进行光学字符识别分析。如果可能，仅用一个单词或短语回答问题。不应包含无关信息。\n',
        'Entity Recognition':'答案应包含问题中提到的实体的识别结果。如果可能，仅用一个单词或短语回答问题。不应包含无关信息。\n',
        'Spatial Awareness':'答案应基于问题中实体之间的空间关系。最好为所有判断提供相应证据。如果图像中没有特定的数值答案，但可以根据其内容进行估计，则可以使用估计结果。仅用一个单词或简短的句子提供答案。不应包含无关信息。\n',
        'Numerical Calculation':'你应根据图像中的信息进行数学计算。你需要估计一些未在图像中直接显示的值以回答问题。你应展示计算过程并输出计算结果。\n',
        'Accurate Numerical Calculation':'你应根据图像中的信息进行数学计算。你需要提供逐步计算过程并获得一个数值结果。\n',
        'Financial Knowledge':'答案应基于金融知识。在 100 字内简要回答问题。答案不应包含与图片无关的内容。\n',
        'Risk Warning':'你应根据图表中的信息和专业金融知识警告投资风险。你的所有论点都需要有事实或理论支持，答案应在 150 字以内。\n',
        'Investment Advice':'你应根据图表中的信息和专业金融知识提供投资建议。你的所有论点都需要有事实或理论支持，答案应在 150 字以内。\n',
        'Explain Reason':'你应根据图表中的信息和专业金融知识提供原因解释。你的所有论点都需要有事实或理论支持，答案应在 150 字以内。\n',
        'Not Applicable':'如果你无法回答，请说“无法回答”，并提供解释。',
    }
    prompt += format_prompt[line['task_category']] if line['task_category'] in format_prompt else ''

# add example
    prompt += """
正确性分数为 0（完全错误）、1、2、3、4、5（完全正确）。
只需完成正确性分数的最后一项空格。
问题 | 真实答案 | 预测结果 | 正确性
--- | --- | --- | --- 
"""
    example_prompt1 = {
        'Image Caption':'图中展示了什么内容？ | 该图像描绘了使用KDJ指标对一项资产进行的技术分析图表。KDJ指标是一种动量振荡器，包括三条线：K线（黄色）：衡量收盘价在特定时间段内的相对位置。D线（青色）：通常是K线的移动平均线。J线（粉色）：代表D线和K线之间的背离。图表中显示的K、D和J的值分别为57.06、63.47和44.22。横轴：时间尺度从去年11月到今年6月中旬。 | 这幅图的主要主题是金融数据的展示。图片展示了各种技术指标，如KJ (9,3,3)，其中K、D和J有特定的值。还有像 “Indicator de” 这样的指标以及一个值为+49.2。此外，一个设置部分提到了多个金融指标，如MACD、KDJ、RSI、BOLL、W&R、DMI、BIAS、ASI、VR、ARBR、DPO、TRIX、New DMA、BBI、MTM、OBV、SAR和EXPMA，并有特定的值。这些元素之间的联系在于它们都是用于分析和理解金融市场的一整套金融指标的一部分。与这些指标相关的值传达了有关市场趋势、动量、波动性和其他特征的特定信息，可帮助交易者和投资者做出明智的决策。 | 3 ',
        'OCR':'2024 年第四季度的经营现金流值是多少？ | 115亿 | 11.50B | 5 ',
        'Entity Recognition':'这份文件包括哪些部分？ | 1.每股收益；2.基于股票的薪酬；3.综合收益；4.衍生品和套期活动；5.信用风险 | 3M公司的归属于净利润、分母、3M普通股股东应占每股收益——基本每股收益、3M普通股股东应占每股收益——稀释每股收益、基于股票的薪酬、综合收益、衍生品和套期活动、信用风险。 | 4 ',
        'Spatial Awareness':'哪个季度显示出最高的机构持股比例？ | 2024年第二季度/2023年第一季度 | 2024年第二季度 | 5 ',
        'Numerical Calculation':'从2022年第一季度到2022年第三季度，机构持股比例变化了多少？ | 2022年第一季度OPTT的机构持股比例约为14%。在2022年第三季度，机构持股比例下降到约8%。因此变化是14%-8%=6%。| 为了计算从2022年第一季度到2022年第三季度机构持股比例的变化，我们用2022年第一季度的比例减去2022年第三季度的比例。2022年第一季度的比例约为12.19%，2022年第三季度的比例约为16.00%。计算如下：16.00%-12.19%=3.81%。从2022年第一季度到2022年第三季度，机构持股比例增加了约3.81%。 | 1 ',
        'Accurate Numerical Calculation':'微软和谷歌的股票价格相差多少？ | 谷歌（GOOGL）的股价是184.49，微软（MSFT）的股价是456.73。所以这两支股票价格的差值是456.73-184.49=272.24。 | 要找出微软（MSFT）和谷歌（GOOGL）股票价格的差异，我们用微软的收盘价减去谷歌的收盘价。微软的收盘价是456.73，谷歌的收盘价是184.49。差值为456.73-184.49=272.24。微软和谷歌的股票价格相差272.24。 | 5 ',
        'Financial Knowledge':'解释什么是经营现金流。 | 经营现金流（OCF）是指公司在一个时期内从经营活动中产生（或消耗）的现金量。 | 经营现金流是指公司核心业务运营所产生的现金流入和流出。它是公司财务健康状况和流动性的关键指标，因为它反映了从日常业务活动中产生现金的能力。经营现金流通过从营业收入中减去经营费用（包括折旧和摊销）来计算。正的经营现金流表明公司正在产生足够的现金来支付费用并投资于增长，而负的经营现金流则表明存在现金短缺，可能需要外部融资或采取成本削减措施。经营现金流是衡量公司从核心业务运营中产生现金能力的指标，通过从营业收入中减去经营费用来计算。 | 2 ',
        'Risk Warning':'基于相对强弱指数（RSI）图表中的信息，可能存在什么潜在风险？ | 分析提示中提到的潜在风险包括接近超买状态、超卖状态、线的交叉、一致性、历史背景和波动性。这些风险表明市场目前正经历高波动性和潜在的趋势变化。建议密切监测 RSI，以发现市场趋势反转的任何迹象。 | 无法回答。提供的图像中不包含相对强弱指数（RSI）图表。因此，无法根据图表中的信息分析潜在风险。RSI 是一种动量振荡器，用于衡量价格变动的速度和变化，以确定超买或超卖情况。没有RSI图表，任何对投资风险的分析都将是推测性的，而不是基于图像中呈现的证据。由于提供的图像中没有RSI图表，所以无法根据图表中的信息分析潜在风险。 | 0 ',
        'Investment Advice':'根据表中的信息，你会给出什么投资建议？ | 基于表和图表中呈现的信息，以下是对英特尔公司（INTC）的详细分析和投资建议：股票概述：当前股价：20.130变化：+0.520（+2.65%） 成交量：48.257K。蜡烛图：开盘价：19.76最高价：20.60最低价：19.75收盘价：20.130（上涨0.52或+2.65%）。技术指标：移动平均线：MA5：30.826 MA20：32.708 MA60：31.284 MA120：35.653 MA250：37.596相对强弱指数（RSI）：RSI6：20.72 股息：0.13（收益率 2.48%） Beta值：1.05。观察结果：趋势：该股票经历了显著的下跌趋势，从近期低点和高成交量峰值可以明显看出。成交量：成交量的增加可能表明投资者兴趣上升或担忧加剧，通常在接近支撑或阻力水平时观察到。RSI水平：RSI值相对较低（某些区间低于30），表明股票可能超卖，暗示潜在的买入机会。移动平均线：股票价格低于短期、中期和长期移动平均线，表明看跌情绪。风险分析：该股票处于下跌趋势，具有显著的看跌情绪。然而，成交量的增加和低RSI表明它可能正在触及支撑水平，并可能准备反弹。短期策略：如果你是短期交易者，你可以考虑进入头寸以利用从超卖状况的潜在反弹。确保设置止损订单以管理风险，以防进一步下跌。 | 要根据图表提供投资建议，需要分析各种指标，如移动平均线、相对强弱指数（RSI）和成交量。例如，如果RSI低于30，可能表明超卖状况，暗示潜在的买入机会。相反，如果RSI高于70，可能表明超买状况，暗示潜在的卖出机会。此外，如果成交量在价格下跌时增加，这可能是卖压增加的迹象，可能表明看跌趋势。投资建议将取决于图表中观察到的具体指标和趋势。例如，如果图表显示看涨趋势且成交量增加，可能建议考虑买入股票。然而，如果图表显示看跌趋势且成交量减少，可能建议考虑卖出股票。需要注意的是，这些是假设的例子，实际的投资决策应基于全面分析，并可能需要咨询财务顾问。 | 1 ',
        'Explain Reason':'从表中的相对强弱指数（RSI）可以推断出哪些潜在风险？ | 该表显示了截至2024年8月16日，几只股票的相对强弱指数（RSI）为100。RSI值为100表明股票处于极度超买状态。这可能带来以下潜在风险：1.Aogo Capital-U（AOGOU） -最新价格：10.97 -RSI：100.00 -风险：高度超买，可能出现价格调整或反转。2.Flag Ship（FSHP） -最新价格：10.00 -RSI：100.00 -风险：高度超买，可能出现价格调整或反转。3.M3-Brigade-U（MBAV.U） -最新价格：10.05 -RSI：100.00 -风险：高度超买，可能出现价格调整或反转。4.ExcelFin-U（XFINU） -最新价格：11.00 -RSI：100.00 -风险：高度超买，可能出现价格调整或反转。5.Four Leaf-U（FORLU） -最新价格：11.03 -RSI：100.00 -风险：高度超买，可能出现价格调整或反转。6.Goldenstone-U（GDSTU） -最新价格：11.30 -RSI：100.00 -风险：高度超买，可能出现价格调整或反转。7.Global Technology-U（GTACU） -最新价格：10.99 -RSI：100.00 -风险：高度超买，可能出现价格调整或反转。总之，表中列出的所有股票的RSI均为100，表明它们处于高度超买状态。这表明这些股票在不久的将来很可能出现下跌或调整，因为投资者可能会抛售股票以实现利润。投资者应考虑到这种风险，并寻找趋势反转的迹象。 | 由于表中所有股票的RSI均为100.00，一个潜在风险是市场可能处于超买状态。RSI为100通常表明资产一直处于强劲的上升势头，但这可能导致反转。例如，在AOGOU Arogo Capital-U的情况下，存在与遵守纳斯达克市值要求相关的风险。这表明，即使股票基于高RSI在短期内看起来很强劲，但仍存在可能影响其未来表现的潜在结构性风险。 | 3 ',
        'Not Applicable':'在相对强弱指标（RSI）图表中，黄色曲线代表什么？ | 在RSI图表中没有黄色曲线，所以答案是不适用。 | 由于所提供的RSI图表中没有黄色曲线，所以无法确定它代表什么。然而，如果有一条黄色曲线，它可能代表RSI框架内的一个特定指标，例如移动平均线或者特定时期的RSI值。例如，它可能是RSI12指标，即12周期的相对强弱指数，用于衡量价格变动的速度和变化，并帮助识别市场中的超买或超卖情况。如果RSI图表中有一条黄色曲线，它可能代表RSI框架内的一个重要指标。 | 0 ',
    }
    prompt += example_prompt1[line['task_category']] if line['task_category'] in example_prompt1 else ''
    example_prompt2 = {
        'Image Caption':'图中展示了什么内容？ | 该图像代表一个股票市场表格，详细列出了各公司的综合信息。各列提供了关键指标，例如：编号：公司在列表中的排名或序号。股票代码：股票代码符号，是用于唯一标识上市公司股票的缩写。公司名称：公司的全名。价格：公司当前的股票价格。变化百分比：股票价格的百分比变化。市值（美元）：截至2024年7月1日公司的市值，以美元表示，表明公司已发行股票的总市场价值。收盘价（美元）：截至2024年7月1日的收盘价。换手率（%）：截至2024年7月1日的换手率百分比，表示股票交易量与流通股数量的关系。成交量（股）：2024年7月1日的股票成交量。美元成交量：2024年7月1日的股票交易美元成交量。GICS行业：公司所属的全球行业分类标准（GICS）行业。开盘价（美元）：2024年7月1日的股票开盘价。 | 这幅图的主要主题是不同公司的财务数据表。每一行代表一家公司，包含股票代码、公司名称、价格、百分比变化、市值、各种财务比率、行业和开盘价等详细信息。这些对象之间的联系在于它们都是特定日期多个公司财务状况综合视图的一部分。像价格和百分比变化这样的元素传达了公司股票的当前价值和变动情况。市值显示了公司在市场中的规模。行业分类有助于了解公司所处的行业领域。这些元素共同为投资者提供了有价值的信息，以便他们分析并对这些公司做出决策。 | 4 ',
        'OCR':'5日、10日、20日、30日和60日简单移动平均线（SMA）的最后一个蜡烛图值是多少？ | 微软公司（MSFT）5日、10日、20日、30日和60日简单移动平均线的最后一个蜡烛图值如下：-SMA5：448.09；-SMA10：444.67；-SMA20：433.29；-SMA30：430.03；-SMA60：420.50。 | 5：448.09，10：444.67，30：430.03，60：420.50。 | 4 ',
        'Entity Recognition':'图表中的蓝色长条代表什么？ | 机构持股比例 | 蓝色长条代表机构持股比例。蓝色长条代表机构持股比例。 | 5 ',
        'Spatial Awareness':'分析从6月1日起K曲线的趋势。 | K曲线在6月初先上升至接近90，接着回落至小于50，最后又再次上升。 | 从6月1日起，K曲线呈现出波动的趋势，先是上升，接着下降，最后又再次上升。从6月1日起，K曲线呈现出波动的趋势。 | 2 ',
        'Numerical Calculation':'分析2021年至2023年确认的减值损失总额的趋势。 | 从2021年到2022年，总减值在上升，从2022年到2023年，总减值在下降。 | 截至2023年、2022年和2021 年6月30日的合并利润表中确认的减值损失总额分别为1800万美元、4300万美元和1000万美元。这一趋势显示从2021年到2022年有所增加，随后在2023年下降。这一趋势显示从2021年到2022年有所增加，随后在2023 年下降。 | 5 ',
        'Accurate Numerical Calculation':'微软和谷歌的股票价格相差多少？ | 谷歌（GOOGL）的股价是184.49，微软（MSFT）的股价是456.73。所以这两支股票价格的差值是456.73-184.49=272.24。 | 为了计算微软（MSFT）和谷歌（GOOGL）之间的股票价格差，我们用微软的收盘价减去谷歌的收盘价。微软的收盘价是456.73美元，谷歌的收盘价是182.99美元。差值是456.73-182.99=273.74美元。微软和谷歌之间股票价格的差值是273.74美元。 | 2 ',
        'Financial Knowledge':'资产负债表提供了关于公司的哪些信息？ | 资产负债表提供了公司在特定时间点的财务状况快照。所提供的资产负债表显示了几个关键指标和趋势：1.债务比率：债务比率在五个季度内被绘制出来。债务比率衡量公司资产中由债务融资的比例。它在2024年第一季度开始为44.85%，在随后的几个季度中略有下降，在2025年第一季度达到36.24%。2.资产：用蓝色条表示，总资产在报告的几个季度中波动，从2024年第一季度到2024年第二季度有所增加，在2024年第三季度有所下降，随后在2024年第四季度再次增加，并在2025年第一季度再次达到峰值。3.负债：用浅蓝色条表示，负债在这段时间内相当稳定，略有下降趋势。4.整体财务健康状况：债务比率下降，同时负债相对稳定，资产变化，这可以被解释为财务健康状况改善的迹象。 | 资产负债表提供了公司在不同季度的债务比率信息。它显示了资产、负债和权益之间的关系。例如，在2024年第二季度，债务比率为56%，并有具体的资产和负债值。这有助于评估公司的财务稳定性和杠杆率。 | 1 ',
        'Risk Warning':'根据图像中的信息，可以识别出哪些潜在风险？ | 内部交易报告显示，公司总裁兼首席执行官黄仁勋（HUANG JEN HSUN）在2024年6月27日同一天进行了多次大规模股票抛售。这种情况为投资者和利益相关者敲响了潜在的警钟。公司高层如此大规模地抛售股票可能表明其对公司未来业绩缺乏信心、可能预知了坏消息，或者认为公司目前的估值相对较高。这可能导致股价波动并可能下跌，因为投资者可能会做出负面反应，将这些抛售行为解读为公司内部出现问题或者即将出现负面情况的迹象。再加上背景信息中强调的公司对狭窄的科技细分市场的依赖以及潜在的政治不确定性，这些内部人士的抛售行为增加了公司股票表现的风险。在这些交易过程中，投资者对首席执行官的稳定性和公司战略的连续性的担忧可能会进一步影响投资者情绪。 | 根据图表中的信息，可以识别出的潜在风险包括首席执行官变更对股价的影响，以及内幕交易的可能性。图表显示在短时间内发生了多次内部交易，这可能引发对潜在利益冲突或内幕交易的担忧。此外，首席执行官的变更可能会潜在地影响公司的业绩和股价，因为投资者对新领导可能会有不同的期望。对于投资者来说，在做出投资决策之前仔细考虑这些风险并进行深入研究是很重要的。 | 0 ',
        'Investment Advice':'基于表格和背景信息，你会给出什么投资建议？ | 基于表格中的信息，以下是对沃尔玛（WMT）的投资分析：1.52周区间：过去一年，该股票的交易价格在49.85美元至76.22美元之间，这表明价格波动范围较大。当前75.60美元的价格接近这个区间的上限，这意味着股票交易价格接近其52周高点。2.日区间：股票的日内价格区间在75.15美元至76.22美元之间，显示出当日有一定的波动性。3.成交量：成交量为1651万股，流动性较高，这对于想要建仓或平仓的投资者来说是个积极因素。4.成交额：股票交易的总金额为12.5亿美元，进一步表明交易活动活跃。5.市盈率（TTM）和市盈率（静态）：市盈率分别为39.09和39.19。这些比率相对较高，这表明与市场平均水平相比，该股票可能被高估。投资者应该考虑公司的增长前景是否能支撑这种估值。6.换手率：换手率为0.38%，这表明换手率相对较低，意味着公司市值中只有一小部分股票是每日交易的。7.市值：沃尔玛的市值为6079.3亿美元，这使其成为大盘股，在市场中地位稳定。8.市净率：市净率为7.20，高于价值投资者通常期望的水平（低于3.0）。这表明从账面价值的角度来看，该股票可能被高估。9.流通股数：有有80.4亿股流通股，这表明市场上有大量股票可供交易。投资建议：1.增长潜力：鉴于市盈率较高，投资者应该评估沃尔玛的增长潜力。如果公司能够维持或加速增长，那么高估值可能是合理的。2.估值担忧：高市盈率和市净率表明股票可能被高估。投资者应该将这些比率与同行业公司进行比较，并考虑沃尔玛的基本面是否支持这种估值。3.市场地位：沃尔玛巨大的市值和高成交量表明其市场地位强劲、流动性好，这对于稳定性和交易便利性来说是积极因素。。4.风险评估：鉴于股票接近其52周高点，短期内上涨空间可能有限。投资者应该警惕潜在的回调。5.长期投资：对于长期投资者来说，沃尔玛强大的市场地位和流动性使其成为相对安全的投资。然而，他们应该关注公司的增长指标和市场状况。总之，虽然沃尔玛是一种稳定且流动性好的投资，但潜在投资者应该仔细考虑其高估值比率，并评估公司的增长前景是否能支撑当前的股价。 | 基于图表中的信息，沃尔玛的股票价格目前为75.60美元，高于其52周低点49.85美元，低于其52周高点76.45 美元。股票贝塔系数为0.50，这表明它的波动性低于大盘。公司的市盈率为39.48，高于其竞争对手好市多、塔吉特和BJ批发俱乐部的市盈率。然而，沃尔玛的市净率为4.71，高于其竞争对手的市净率，这表明相对于账面价值，该股票可能被高估。总体而言，图表显示沃尔玛的股票可能被高估，投资者在做投资决策时应该考虑到这一点。 | 0 ',
        'Explain Reason':'过去两年公司员工减少的主要原因是什么？ | 在过去两年中，公司员工减少的主要原因可以总结如下：1. 全球宏观经济因素和中国严格的 “动态清零” 政策影响了消费者支出和公司的增长。2. 公司一直在减少员工人数以提高成本效率并适应经济放缓。这些原因有以下证据支持：1. 净利润在过去两年中表现出显著的波动性，在2022年3月31日急剧下降至约65000，随后在2023年3月31日恢复至约93000，并在2024年6月30日略有下降至约86000。这表明公司在维持盈利能力方面面临挑战。（参考：背景）2. 阿里巴巴集团在2023年期间裁员约20000人，这增加了从硅谷到杭州的全球科技公司的一系列裁员。该公司裁员的速度相当稳定，因为它在2022年期间也减少了大致相同数量的员工。（参考：阿里巴巴在2023年进一步裁员20000人）3. 公司的增长受到全球宏观经济因素和中国严格的 “动态清零” 政策的阻碍，封锁抑制了消费者支出。阿里巴巴在5月份表示，它将采取 “更有纪律” 的支出方式，并缩减在不能产生长期价值的领域的开支。这种转变与北京的激励措施一致，标志着与这家电子商务巨头过去积极广泛的市场抢占策略的重大变化。（参考：阿里巴巴在2022年裁员19000人以提高成本……）总之，过去两年公司员工的减少可归因于经济因素和公司为提高成本效率并适应经济放缓所做的努力的结合。 | 根据图表中的信息，过去两年公司员工减少的主要原因可以归因于公司为提高成本效率并适应全球经济降温所做的努力。在2022年，阿里巴巴集团控股有限公司减少了约19000名员工，在2023年，该公司在这一年中裁员约20000人。公司的增长受到全球宏观经济因素和中国严格的 “动态清零” 政策的阻碍，封锁抑制了消费者支出。阿里巴巴在5月份表示，它将采取 “更有纪律” 的支出方式，并缩减在不能产生长期价值的领域的开支。这种转变与北京的激励措施一致，标志着与这家电子商务巨头过去积极广泛的市场抢占策略的重大变化。阿里巴巴在其最新报告中仍拥有239740名员工，仍然是中国最大的私营部门雇主之一。它报告的销售额和利润超出了分析师的平均预期，其股价因此飙升。尽管如此，它在12月份的裁员比9月份有所增加，因此该公司似乎仍在调整其规模以应对经济放缓——即使国内外经济复苏的迹象越来越明显。 | 4 ',
        'Not Applicable':'分析家乐福的销售收入趋势。 | 根据图表无法回答这个问题。因此，答案是不适用。 | 不适用。所提供的图像中不包含图表或任何关于十年期间收入的可视化表示。因此，无法根据图像中给出的信息确定收入是如何变化的。图像中的信息不足以回答关于十年间收入变化的问题。 | 5 ',
    }
    prompt += example_prompt2[line['task_category']] if line['task_category'] in example_prompt2 else ''
    gpt4_prompt = prompt + '\n' + ' | '.join(
        [question, gt.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> '), prediction, ''])
    return gpt4_prompt

def MMfin_CN_auxeval(model, line):    
    def int_cvt(s):
        try:
            return int(s)
        except ValueError:
            return find_last_int_in_range(s)

    prompt = build_mmfin_gpt4_prompt_cn(line)
    log = ''
    retry = 5
    for i in range(retry):
        mes = [{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': prompt,
                }, {
                    'type': 'image_url',
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(osp.join(LMUDataRoot(), 'images/MMfin_CN', line['image_path']))}"
                    }
                }],
            }]
        output = model.generate(mes, temperature=i * 0.5)
        score = int_cvt(output.choices[0].message.content)
        if score is None:
            log += f'Try {i}: output is {output}, failed to parse.\n'
        elif score < 0 or score > 5:
            log += f'Try {i}: output is {output}, invalid score: {score}.\n'
        else:
            log += f'Try {i}: output is {output}, parse succeed'
            return dict(log=log, score=score)
    print(f'All {retry} retries failed.')
    print('log: ',log)
    log += f'All {retry} retries failed.\n'
    return dict(log=log, score=0.0)


def MMfin_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    lt = len(data)
    cate_list = []
    cate2_list = []
    cate3_list = []
    for i in range(lt):
        item = data.iloc[i]
        cate = item['task_category']
        cate2 = item['image_type']
        cate3 = item['image_style']
        if cate not in cate_list:
            cate_list.append(cate)
        if cate2 not in cate2_list:
            cate2_list.append(cate2)
        if cate3 not in cate3_list:
            cate3_list.append(cate3)
        grade = float(item['score'])
        
        tot['Overall'] += 1
        tot[cate] += 1
        tot[cate2] += 1
        tot[cate3] += 1
        score['Overall'] += grade
        score[cate] += grade
        score[cate2] += grade
        score[cate3] += grade

    res = defaultdict(list)
    res2 = defaultdict(list)
    res3 = defaultdict(list)
    cate_list.append('Overall')
    cate2_list.append('Overall')
    cate3_list.append('Overall')
    for k in cate_list:
        res['Category'].append(k)
        res['tot'].append(tot[k])
        res['acc'].append(score[k] / tot[k] * 20)
    for v in cate2_list:
        res2['Category'].append(v)
        res2['tot'].append(tot[v])
        res2['acc'].append(score[v] / tot[v] * 20)
    for w in cate3_list:
        res3['Category'].append(w)
        res3['tot'].append(tot[w])
        res3['acc'].append(score[w] / tot[w] * 20)
    res = pd.DataFrame(res)
    res2 = pd.DataFrame(res2)
    res3 = pd.DataFrame(res3)
    
    total_score = defaultdict(lambda: 0)
    for key in tot:
        total_score[key] = score[key] / tot[key] * 20
    return res, res2, res3, total_score