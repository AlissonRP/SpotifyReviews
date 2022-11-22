#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utils.utils import my_wc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



#%%
df = pd.read_csv("data/reviews.csv")

#%%


def total_variable(df, var):
    data = (
        df.groupby(var, as_index=False)["Rating"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    data = data.assign(PROPORCAO=data["sum"] / data["count"])
    return data


#%%


grouped_data = df.groupby("Rating", as_index=False).size()
grouped_data = grouped_data.assign(
    Proporcao=grouped_data["size"] / grouped_data["size"].sum()
)
sns.barplot(data=grouped_data, x="Rating", y="Proporcao", palette="viridis")
plt.xlabel("Avaliação")
plt.ylabel("Proporção")
plt.title("Proporcão das Avaliações")


#%%

df["Time_submitted"] = pd.to_datetime(df["Time_submitted"])

#%%
df["month"] = df["Time_submitted"].apply(lambda x: x.month)
df["hour"] = df["Time_submitted"].apply(lambda x: x.hour)
df["day"] = df["Time_submitted"].apply(lambda x: x.day)

#%%
df.groupby("month", as_index=False)["Rating"].mean()


def time_plot(df, var):
    total_hour = total_variable(df, var)
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("teste")
    sns.lineplot(data=total_hour, x=var, y="count", ax=ax[0])
    sns.lineplot(data=total_variable(df, "day"), x="day", y="mean", ax=ax[1])
    fig.show()


#%%
total_hour = total_variable(df, "day")
sns.lineplot(data=total_hour, x="day", y="count")

#%%
time_plot(df, "day")


#%%
# sns.lineplot(data=total_variable(df, 'month'), x='month', y='count')
bp = sns.barplot(data=total_variable(df, "month"), x="month", y="count")
# plt.bar_label(bp.containers[0])
bp = sns.lineplot(data=total_variable(df, "month"), x="month", y="count")


#%%
# use the containers of the barplot to generate the labels

sns.lineplot(data=total_variable(df, "day"), x="day", y="mean")

#%%
top_50 = df["Review"].str.split(expand=True).stack().value_counts()[:50]


#%%
wc = WordCloud(
    width=1500, height=1500, background_color="white", max_words=500
).generate(" ".join(df["Review"]))
plt.axis("off")
plt.imshow(wc)

#%%
from utils.utils import my_wc

my_wc(df, "podre")

#%%
word_tokenize(df["Review"][0])
#%%


df["Review_words"] = [len(x.split()) for x in df["Review"]]


#%%
rating = df["Rating"]
df["Sentiment"] = np.where(
    (1 <= rating) & (rating <= 2),
    "negativo",
    np.where((4 <= rating) & (rating <= 5), "positivo", "neutral"),
)
#%%

df[df['Sentiment'] == 'neutral'][['Review_words']].describe()

#%%
sns.boxplot(data=df[df["Sentiment"] == 'positive'], x="Review_words")


#%%
sns.lineplot(data=df[df['Review_words']< 180], y='Review_words', x='day', hue='Sentiment')


#%%
stop_words = set(stopwords.words("english"))
stop_words.update(["app", "song", "Spotify", "music", "spotify", "App", "app", "songs"])
#%%
def removing_sw(word):
    return [w for w in word.split() if w not in stop_words]

#%%
df['Review_wsw'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
import re

#%%
import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

#%%
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

#%%
df['Review_wsw'] = df['Review_wsw'].apply(lemmatize_text)
df['Review_wsw'] = df['Review_wsw'].apply(lambda x: " ".join(x))

#%%

#%%
my_wc(df,  Filter = 'podre', var = 'Review_wsw')

#%%
my_wc(df,  Filter = 'podre', var = 'Review_wsw')

#%%
df['Review_wsw'][4]

#%%
data = df[df["Rating"] == 5]['Review_wsw']
wc = WordCloud(
        width=1000, height=1000, background_color="white", max_words=500
    ).generate(" ".join(data))
plt.imshow(wc)

#%%

word = df[df["Rating"] == 1]['Review'].apply(word_tokenize)

#%%
from nltk import ngrams, FreqDist

#%%
FreqDist(ngrams(word, 1))

#%%
text1 = '''Seq Sentence 
1   Let's try to be Good.
2   Being good doesn't make sense.
3   Good is always good.'''


data = df[df['Sentiment'] != 'neutral']
## MODELAGEM
X = data['Review_wsw']
y = data['Sentiment']

#%%
plt.pie(x=y.value_counts(), labels=y.unique(), autopct='%.0f%%')
plt.title("Proporção dos Sentimentos")

#%%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
vectorizer = CountVectorizer(max_df=0.8)
le = LabelEncoder()
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)


#%%

vectorizer.fit(X_train)
X_train_count = vectorizer.transform(X_train)
X_test_count = vectorizer.transform(X_test)

#%%
from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier(random_state=19, n_estimators = 105, min_samples_split =  12, min_samples_leaf = 1)
rf1.fit(X_train_count, y_train)
rf_cv_accuracy = rf1.score(X_test_count, y_test)
print(f'Random Forest Classifier on Count Vectors: {rf_cv_accuracy}')
#%%


#%%

predictions = rf1.predict(X_test_count)
cm = confusion_matrix(y_test, predictions, labels=rf1.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf1.classes_)
disp.plot()
disp.ax_.set(xlabel='Predito', ylabel='Verdadeiro');



#%%
def my_predict(text):
    teste = vectorizer.transform(pd.Series(text))
    return rf1.predict(teste)

#%%
my_predict("Great music service, the audio is high quality and the app is easy to use. Also very quick and friendly support.")

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2,3))
tfidf.fit(X_train)
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


#%%
rf2 = RandomForestClassifier()
rf2.fit(X_train_tfidf, y_train)
rf_tfidf = rf2.score(X_test_tfidf, y_test)
print(f'Random Forest Classifier on TF-IDF Vectors: {rf_tfidf}')

#%%


X_test.reset_index(drop=True)[4]


#%%
y_test.reset_index(drop=True)[4]
# %%
# %%
from sklearn.linear_model import LogisticRegression

rl = LogisticRegression(random_state=0, penalty='none', max_iter=10000)

#%%
rl.fit(X_train_count, y_train)
rl_cv = rl.score(X_test_count, y_test)
print(f'Logistic Regression Classifier on Count Vectors: {rl_cv}')

#%%
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(1,2)
RocCurveDisplay.from_estimator(rf1, X_test_count, y_test, ax=ax[0])
RocCurveDisplay.from_estimator(rl, X_test_count, y_test, ax=ax[1]);


#%%


df[df['Review_words']<4][['Review', "Rating"]].head(5).to_markdown()