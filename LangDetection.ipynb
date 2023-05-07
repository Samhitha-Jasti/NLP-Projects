{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ff0d9ee4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import re          #regular expression\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.simplefilter(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5455028e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Text</th>\n",
       "      <th>Language</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Nature, in the broadest sense, is the natural...</td>\n",
       "      <td>English</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>\"Nature\" can refer to the phenomena of the phy...</td>\n",
       "      <td>English</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>The study of nature is a large, if not the onl...</td>\n",
       "      <td>English</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Although humans are part of nature, human acti...</td>\n",
       "      <td>English</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>[1] The word nature is borrowed from the Old F...</td>\n",
       "      <td>English</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>[2] In ancient philosophy, natura is mostly us...</td>\n",
       "      <td>English</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>[3][4] \\nThe concept of nature as a whole, the...</td>\n",
       "      <td>English</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>During the advent of modern scientific method ...</td>\n",
       "      <td>English</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>[5][6] With the Industrial revolution, nature ...</td>\n",
       "      <td>English</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>However, a vitalist vision of nature, closer t...</td>\n",
       "      <td>English</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                Text Language\n",
       "0   Nature, in the broadest sense, is the natural...  English\n",
       "1  \"Nature\" can refer to the phenomena of the phy...  English\n",
       "2  The study of nature is a large, if not the onl...  English\n",
       "3  Although humans are part of nature, human acti...  English\n",
       "4  [1] The word nature is borrowed from the Old F...  English\n",
       "5  [2] In ancient philosophy, natura is mostly us...  English\n",
       "6  [3][4] \\nThe concept of nature as a whole, the...  English\n",
       "7  During the advent of modern scientific method ...  English\n",
       "8  [5][6] With the Industrial revolution, nature ...  English\n",
       "9  However, a vitalist vision of nature, closer t...  English"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"Language Detection.csv\")\n",
    "data.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9ed37f9e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "English       1385\n",
       "French        1014\n",
       "Spanish        819\n",
       "Portugeese     739\n",
       "Italian        698\n",
       "Russian        692\n",
       "Sweedish       676\n",
       "Malayalam      594\n",
       "Dutch          546\n",
       "Arabic         536\n",
       "Turkish        474\n",
       "German         470\n",
       "Tamil          469\n",
       "Danish         428\n",
       "Kannada        369\n",
       "Greek          365\n",
       "Hindi           63\n",
       "Name: Language, dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[\"Language\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "543a824a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "17"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[\"Language\"].nunique()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "47a85562",
   "metadata": {},
   "source": [
    "There are 17 languages in dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "bd09e4b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Separating Independent and Dependent features\n",
    "X = data[\"Text\"]\n",
    "y = data[\"Language\"]\n",
    "\n",
    "#Label Encoding to convert it into a numerical form\n",
    "#For example, English as 3.. similarily all languages are given a unique label\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "le = LabelEncoder()\n",
    "y = le.fit_transform(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8158aeb1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([3, 3, 3, ..., 9, 9, 9])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "8af9a072",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_list = []\n",
    "\n",
    "# iterating through all the text\n",
    "for text in X:         \n",
    "    text = re.sub(r'[!@#$(),n\"%^*?:;~`0-9]', ' ', text)      # removing the symbols and numbers\n",
    "    text = re.sub(r'[[]]', ' ', text)   \n",
    "    text = text.lower()          # converting the text to lower case\n",
    "    data_list.append(text)       # appending to data_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5303d777",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10337, 34937)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Bag of Words [ converting text into numerical form by creating a Bag of Words model using CountVectorizer.]\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "cv = CountVectorizer() # tokenize a collection of text documents\n",
    "X = cv.fit_transform(data_list).toarray()\n",
    "X.shape # (10337, 39419)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7526ca32",
   "metadata": {},
   "source": [
    "As the model can't understand textual data, all the data is sent in numerical format to the model after preprocessing data(removing symbols, spaces).\n",
    "The textual data need to be vectorized.\n",
    "\n",
    "Countvectorizer Tokenization(Tokenization Means Breaking Down A Sentence Into Words By Performing Preprocessing Tasks Like Converting All Words To Lowercase, Removing Special Characters, Etc.\n",
    "\n",
    "Bag of Words(BOW) model represents text that describes the occurence of words within a document. It follows binary representation. 1 for frequent occurence of a word and 0 if word is not occuring"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "88fe3a72",
   "metadata": {},
   "outputs": [],
   "source": [
    "#train-test split\n",
    "from sklearn.model_selection import train_test_split\n",
    "x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "51c360b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MultinomialNB()"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Model Training\n",
    "from sklearn.naive_bayes import MultinomialNB  #classifier is suitable for classification with discrete features\n",
    "model = MultinomialNB()\n",
    "model.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "42ba1c18",
   "metadata": {},
   "source": [
    "The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f8585570",
   "metadata": {},
   "outputs": [],
   "source": [
    "#predict output for test dataset\n",
    "y_pred = model.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f72e4139",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
    "ac = accuracy_score(y_test, y_pred)\n",
    "cm = confusion_matrix(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "37c606e3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy is : 0.9796905222437138\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy is :\",ac)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "adcf5cfd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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nBpC+MrMHqD0+Lv3VPSgssNtDQawjKbEYg2rFZmfV0CHnOwdn+e0b0dBwYpqPWdtobHbd3TudQRrlq5k85qNAMybOJMDvvstZr/4b574cek/xZad2nLIScdGmQn4/rpTW+I893luA999ats3eR87730pl8ajVHkQb5OqmJk1M7MWgIUQvgIIIWym9DToFQohjAghdAkhdKnuygbAzFmz6dy5Ex07diArK4tBgwbyyvhJ1X7e2kBjt+9Yv2wV+x97MFn16gDQuefhrPw8n4YtmgClfy066apzmP73N6PMBHx/3aktcZ77PLeB7z617Zu8j533PkmueDMcTYH/UnqCj2BmbUMIy8tODFLRST9Sori4mCFDb2XCq8+SEYsx+smx5OZ+mq7Fx/XM049y4gk9aNmyOYu/mMXtdzzAqNFjos4CNHbV4a1t6eyFzH1tOle9ejclRcUUzFvMjOfeovuPvkePi08FYO7rM/nv8+9G1rid5687tSXOc5/nNvDdp7bEefs5UZ73sfPeJ8lliWwvZ2YNKD3d+aJ4982s0752zRmJxHF99olRJ1Tp/mXRr7CIiIgkomhbftr+IF4dGy47NW2/Hzd5/I3IxyShM42HELYAcVc2RERERESkdktohUNERERERBITtNO4iIiIiIhIcmiGQ0REREQknTTDISIiIiIikhya4RARERERSaeSqAPSSzMcIiIiIiKSMprhEBERERFJIx2lSkREREREJEk0wyEiIiIikk6a4RAREREREUkOzXCIiIiIiKRTLTtKlVY4RNLs/mXvRp1QpUHtukWdUKlxBTOiThAREZG9pBUOEREREZE00lGqREREREREkkQrHCIiIiIikjLapEpEREREJJ1q2U7jmuEQEREREZGU0QyHiIiIiEgaaadxERERERGRJNEMh4iIiIhIOmkfDhERERERkeTQDIeIiIiISBoFzXD41Pe0PsybO4UFudO44foro87Zjec+z20jRwxjWd4cZn84OeqUCnkeO49tDZo0YMhj1/PA5D9x/+Q/cfB3DuHqR67l7gl/5O4Jf2T4tL9y94Q/Rp3pcuy289wGvvs8t4HvPrUlznOf5zbw3yfJYyGkdi/5zDrtq72AWCzG/HlT6XfGheTlFfD+exO46OIrmD//s2QkVpvnPs9tAL17dWfTps2MGjWcY449JeqcnXgeu1S2DWrX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\n",
      "text/plain": [
       "<Figure size 1080x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,10))\n",
    "sns.heatmap(cm, annot = True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "4dbbc541",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Predicting with some more data\n",
    "\n",
    "def predict(text):\n",
    "     x = cv.transform([text]).toarray() # converting text to bag of words model (Vector)\n",
    "     lang = model.predict(x) # predicting the language\n",
    "     lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value\n",
    "     print(\"The langauge is in\",lang[0]) # printing the language"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "0e8ec3f3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The langauge is in English\n"
     ]
    }
   ],
   "source": [
    "predict('People are awesome')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "609cf6e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The langauge is in Malayalam\n"
     ]
    }
   ],
   "source": [
    "predict('നൽകുന്നു')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "35af08e8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The langauge is in Hindi\n"
     ]
    }
   ],
   "source": [
    "predict('अरे आप कैसे हैं')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "fcf3fc8c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The langauge is in English\n"
     ]
    }
   ],
   "source": [
    "predict('Bugün nasılsın') #turkish"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "8c50005c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The langauge is in English\n"
     ]
    }
   ],
   "source": [
    "predict('ನೀವು ಇಂದು ಹೇಗಿದ್ದೀರಿ')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ce1fe747",
   "metadata": {},
   "source": [
    "As the above model is not predicting the right languages based on text, we will try a different model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e9e801c3",
   "metadata": {},
   "source": [
    "# Using FastText"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "1edced71",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Defaulting to user installation because normal site-packages is not writeable\n",
      "Collecting fasttext\n",
      "  Using cached fasttext-0.9.2.tar.gz (68 kB)\n",
      "Requirement already satisfied: pybind11>=2.2 in c:\\users\\asjas\\appdata\\roaming\\python\\python39\\site-packages (from fasttext) (2.10.4)\n",
      "Requirement already satisfied: setuptools>=0.7.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from fasttext) (61.2.0)\n",
      "Note: you may need to restart the kernel to use updated packages.Requirement already satisfied: numpy in c:\\programdata\\anaconda3\\lib\\site-packages (from fasttext) (1.21.5)\n",
      "Building wheels for collected packages: fasttext\n",
      "  Building wheel for fasttext (setup.py): started\n",
      "  Building wheel for fasttext (setup.py): still running...\n",
      "  Building wheel for fasttext (setup.py): finished with status 'done'\n",
      "  Created wheel for fasttext: filename=fasttext-0.9.2-cp39-cp39-win_amd64.whl size=227412 sha256=32b53bf33c30e7e374d02ba992dc06a31c14f7bc45ac5ce1ab35e91c70300aa8\n",
      "\n",
      "  Stored in directory: c:\\users\\asjas\\appdata\\local\\pip\\cache\\wheels\\64\\57\\bc\\1741406019061d5664914b070bd3e71f6244648732bc96109e\n",
      "Successfully built fasttext\n",
      "Installing collected packages: fasttext\n",
      "Successfully installed fasttext-0.9.2\n"
     ]
    }
   ],
   "source": [
    "pip install fasttext"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3cb5bf85",
   "metadata": {},
   "outputs": [],
   "source": [
    "import fasttext as ft"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6a02e968",
   "metadata": {},
   "source": [
    "FastText is an open-source and lightweight library developed by Facebook AI Research. It efficiently handles large datasets, allowing for the classification of billions of words in milliseconds. \n",
    "\n",
    "FastText also provides pre-trained word vectors for 157 languages, allowing for transfer learning and quick integration into NLP models. With its user-friendly interface and high performance, FastText has become a popular choice for NLP tasks such as sentiment analysis, spam filtering, and language identification."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "499ae4f4",
   "metadata": {},
   "source": [
    "There are several reasons why FastText is fast while maintaining performance. Some of them are:\n",
    "\n",
    "- It is implemented in C++\n",
    "- It allows you to use multiprocessing during training\n",
    "- Based on a simple neural network architecture"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "7d817177",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.\n"
     ]
    }
   ],
   "source": [
    "model = ft.load_model('lid.176.ftz')\n",
    "#lid.176.ftz: A model trained on text of 176 different languages. \n",
    "#This is the most common pre-trained model available for language detection."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "51e9468e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(('__label__en',), array([0.96859217]))\n"
     ]
    }
   ],
   "source": [
    "language = model.predict('This is a sentence in English')\n",
    "# Print predicted language\n",
    "print(language)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d8719ea3",
   "metadata": {},
   "source": [
    "af – Afrikaans\tlv – Latvian\tho – Hiri Motu\n",
    "am – Amharic\tmg – Malagasy\thz – Herero\n",
    "an – Aragonese\tmi – Maori\tii – Sichuan Yi\n",
    "ar – Arabic\tmk – Macedonian\tinh – Ingush\n",
    "as – Assamese\tml – Malayalam\tjbo – Lojban\n",
    "az – Azerbaijani\tmn – Mongolian\tkl – Kalaallisut\n",
    "be – Belarusian\tmr – Marathi\tks – Kashmiri\n",
    "bg – Bulgarian\tms – Malay\tku-Latn – Central Kurdish (Latin)\n",
    "bn – Bengali\tmt – Maltese\tkv – Komi\n",
    "br – Breton\tmy – Burmese\tlb – Luxembourgish\n",
    "bs – Bosnian\tne – Nepali\tlg – Ganda\n",
    "ca – Catalan\tnl – Dutch\tmh – Marshallese\n",
    "ceb – Cebuano\tno – Norwegian\tmi – Māori\n",
    "co – Corsican\tny – Chichewa\tmrj – Hill Mari\n",
    "cs – Czech\tpa – Punjabi\tnah – Nahuatl\n",
    "cy – Welsh\tpl – Polish\tnap – Neapolitan\n",
    "da – Danish\tpt – Portuguese\tnb – Norwegian Bokmål\n",
    "de – German\tro – Romanian\tnn – Norwegian Nynorsk\n",
    "el – Greek\tru – Russian\tno – Norwegian\n",
    "en – English\tsd – Sindhi\toc – Occitan\n",
    "eo – Esperanto\tsi – Sinhala\tos – Ossetian\n",
    "es – Spanish\tsk – Slovak\tpi – Pali\n",
    "et – Estonian\tsl – Slovenian\tps – Pashto\n",
    "eu – Basque\tsm – Samoan\tqu – Quechua\n",
    "fa – Persian\tsn – Shona\trm – Romansh\n",
    "fi – Finnish\tso – Somali\trn – Rundi\n",
    "fr – French\tsq – Albanian\trw – Kinyarwanda\n",
    "fy – Western Frisian\tsr – Serbian\tsc – Sardinian\n",
    "ga – Irish\tst – Sotho\tse – Northern Sami\n",
    "gd – Scottish Gaelic\tsu – Sundanese\tsg – Sango\n",
    "gl – Galician\tsv – Swedish\ttk – Turkmen\n",
    "gu – Gujarati\tsw – Swahili\ttlh – Klingon\n",
    "ha – Hausa\tta – Tamil\ttn – Tswana\n",
    "haw – Hawaiian\tte – Telugu\tto – Tongan\n",
    "hi – Hindi\ttg – Tajik\ttw – Twi\n",
    "hmn – Hmong\tth – Thai\tty – Tahitian\n",
    "hr – Croatian\ttl – Tagalog\tug – Uighur\n",
    "ht – Haitian Creole\ttr – Turkish\tve – Venda\n",
    "hu – Hungarian\tuk – Ukrainian\tvo – Volapük\n",
    "hy – Armenian\tur – Urdu\twa – Walloon\n",
    "id – Indonesian\tuz – Uzbek\two – Wolof\n",
    "ig – Igbo\tvi – Vietnamese\txog – Soga\n",
    "is – Icelandic\txh – Xhosa\tyue – Cantonese\n",
    "it – Italian\tyi – Yiddish\tza – Zhuang\n",
    "iw – Hebrew\tyo – Yoruba\tace – Acehnese\n",
    "ja – Japanese\tzh – Chinese\tach – Akan\n",
    "jw – Javanese\tzu – Zulu\tady – Adyghe\n",
    "ka – Georgian\tak – Akan\taln – Gheg Albanian\n",
    "kk – Kazakh\tbh – Bihari\talt – Southern Altai\n",
    "km – Khmer\tbi – Bislama\tanp – Angika\n",
    "kn – Kannada\tbm – Bambara\tarn – Mapudungun\n",
    "ko – Korean\tchr – Cherokee\tarq – Algerian Arabic\n",
    "ku – Kurdish\tdv – Divehi\tast – Asturian\n",
    "ky – Kyrgyz\tdz – Dzongkha\tav – Avar\n",
    "la – Latin\tee – Ewe\tazb – South Azerbaijani\n",
    "lb – Luxembourgish\tff – Fulah\tba – Bashkir\n",
    "lo – Lao\tgv – Manx\tbar – Bavarian\n",
    "lt – Lithuanian\thak – Hakka Chinese\tbbc – Batak Toba\n",
    "bcl – Central Bicolano\tbe-tarask – Belarusian (Taraškievica)\t"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "3495f66d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(('__label__tr',), array([0.98389137]))"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict('Bugün nasılsın')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "e4219d5b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(('__label__kn',), array([0.99934846]))"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict('ನೀವು ಇಂದು ಹೇಗಿದ್ದೀರಿ')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8edcdd5d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
