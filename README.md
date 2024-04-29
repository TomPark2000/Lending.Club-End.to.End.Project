# Lending.Club-End.to.End

LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.

The main objective was to create a model that predicts whether a borrower will pay back their loan given historical data. Through this, LendingClub can assess whether a potential customer is likely to pay back the loan.

EDA and Feature Engineering:


**Initial EDA**
I visualized several the relationships in order to get a better understanding of the features and the correlations of those features.

Since we will be attempting to predict loan status, I started with a count plot.:


I then wanted to see if there was a relationship between the loan status and the loan amount. It seems that there's a slight increase in likelihood of the loan not being paid off if it's higher, which makes sense


I dove deeper by looking at the summary statistics of the loan status and loan amount. This confirmed the initial findings.


Looking at correlations
To get a sense of the correlations for all the numeric columns, I created the heatmap below. The lighter the color, the more positevly correlated the two features are. There are very strong correlations between features like "pub_rec_bankruptcies", "open_acc", "total_acc", "int_rate", "annual_income", and more. These all make logical sense given the context and description. 



For example, there's an almost perfect correlation with the "installment" and "loan_amnt" feature, which makes sense given that “installment” is the monthly payment owed by the borrower if the loan originates. I visualized these two features below:


**grade and 
Next I wanted to visualize the “grade” (LC assigned loan grade) and the rate of the loans being paid off. As expected, the better grades (ex. A) have a much higher likelihood of being paid off compared to lower grades (ex. G). 


I dug deeper into the F and G subgrades to see if there was a clear distinction between the subgrades of the grades. There was a general trend but nothing too different. 


Feature Engineering:
Extract first 3 zipcode numbers from address feature -> Grouped the first 3 letters of the zip code
Changed the home_ownership feature -> Most are in mortgage, rent, or own. Since there's so few in "none" and "any", we'll replace them into "other"

Categorized annual_inc -> there are too many unique incoms so I categorized the income by classes from the Census Bureau's 2022 report for Income in the United States, with Lower class: less than or equal to $30,000, Lower-middle class:  30,001– 58,020, Middle class:  58,021– 94,000, Upper-middle class:  94,001– 153,000, and Upper class: greater than $153,000
Dti -> several outliers that are unrealistic or too high. For ex. 9999, +6000, etc. So I deleted the top 0.05% of the values
Extracted the year from earliest_cr_line then categorized by decade
