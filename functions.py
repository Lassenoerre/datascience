
#------------------------------------------------#
#                    SCRAPING                    #
#------------------------------------------------#

def get_links_to_scrape(start_year, end_year):

    """
    This function takes a starting year and an ending year and crawls through
    CNBC's SiteMap structure to find all links to articles that are written
    in the given time period.
    """

    # Import of the required packages
    import requests
    from bs4 import BeautifulSoup, SoupStrainer

    # First, we define every month and year that should be scraped.
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    years = [year for year in range(start_year, end_year+1)]

    # In order to scrape the links faster, we make use of SoupStrainer
    # to only parse the relevant part of the site: ALl the article links.
    strainer = SoupStrainer('a', class_ = 'SiteMapArticleList-link')

    # We instantiate an empty list that we can store all the links in.
    article_links = []

    # For every day, in every month in every year, construct a link
    # that should be scraped. We do this, because CNBC's link structure
    # in their archive is given af .../site-map/YEAR/MONTH/DAY_OF_MONTH.
    for year in years:
        for month in months:
            for day in range(1,32):
                link = f'https://www.cnbc.com/site-map/{year}/{month}/{day}/'

                # Take the generated link and load it in through requests,
                # then take the contents and convert it to a string.
                page = str(requests.get(link).content)

                # CNBC load in their CSS directly into the HTML, with usually
                # would make it very heavy to scrape. We can fix this, simply
                # by cutting remove the entire <head></head> part of the html.
                page = page[str(page).find('body'):]

                # Now, pass the remaining content through BeautifulSoup's parser
                # and apply the strainer. This means it will only parse link
                # elements with the class of SiteMapArticleList-Link and it
                # will only search through body-section of the document.
                soup = BeautifulSoup(page, parse_only = strainer)

                # Now we make use of the find_all function in BeautifulSoup
                # that simply generates a list of codeblocks that match the
                # query. This means we get a list of links, and we store the
                # href (actual links) in our instantiated list from before.
                links = soup.find_all('a', class_='SiteMapArticleList-link')
                if links != []:
                    for a_link in links:
                        article_links.append(a_link['href'])

    # Return the final list of article links that we want to scrape.
    return article_links


def scrape_cnbc_articles(list_of_links):

    """
    This function take a list of article links and attempts to scrape
    them according to CNBC's HTML structure. It will return a pandas
    DataFrame with the scraped data.
    """

    # Import of the required packages
    import requests
    from bs4 import BeautifulSoup, SoupStrainer
    import pandas as pd

    # First, we instantiate an empty list, that we can append the scraped
    # data to. Then we instantiate an index, which enables us to follow the
    # scrapers progress, and lastly, we instantiate a request Session, which
    # minimizes our load time per request by little bit, but that time will
    # accumulate for all our links and should have quite an impact.
    df = []
    index = 0
    request = requests.Session()

    # For every link in the list, get the source code and add 1 to the index.
    for link in list_of_links:
        page = request.get(link)
        index += 1

        # If the pages HTTP request Code is 200 (Meaning "I loaded the site
        # correctly, and the server sent me the data correctly"), then
        # attempt to scrape the site.
        if page.status_code == 200:
            try:

                # First, we convert the page content to a string, so we can
                # remove most of the irrelevant HTML. We remove everything
                # before a box with the class "MainContent"
                page = str(page.content)
                page = page[page.find('<div id="MainContent"'):]

                # Now we parse the remaining content of the page into
                # BeautifulSoup and try to extract the text of:
                soup_link = BeautifulSoup(page)

                # The Header-1 tag with a class of 'ArticleHeader-headline'
                title = soup_link.find('h1', class_='ArticleHeader-headline').get_text()

                # The div (box) tag with a class of 'ArticleBody-articleBody'
                article = soup_link.find('div', class_='ArticleBody-articleBody').get_text()

                # The date generated from the link. The link will always
                # contain the date if it is an article. We save it as
                # DD/MM/YYYY.
                date = f'{link[29:31]}-{link[26:28]}-{link[21:25]}'

                # The link that have a class of 'ArticleHeader-eyebrow',
                # which is their article topic.
                topic = soup_link.find('a', class_='ArticleHeader-eyebrow').get_text()

                # If they are all successfully gathered, we append it all
                # into our list called df.
                df.append([title, topic, date, article, link])

                # If successful, print the progress as well as the link.
                print(f'({index}/{len(list_of_links)}) : {link}')
            except:
                # If we get a status code 200, but somehow some of the elements
                # wasn't there, then skip the entire article. This ensures that
                # we get a dataset without missing variables.
                print(f'({index}/{len(list_of_links)}) : Skipped')
        else:
            # If we didn't get a status code 200 (Meaning something went wrong
            #  in the loading of the page), then skip the article.
            print(f'({index}/{len(list_of_links)}) : Skipped')

    # Lastly, we return a dataframe that contains all of our scraped articles.
    return pd.DataFrame(df)


def collect_dataset(datasets_path):

    """
    This function takes all the files in a folder and tries to concatenate them
    into one dataframe. This is of course only possible if your datasets is
    the only csv-files in the folder and if they have the same data-structure.
    We made this, because we scraped the links over several sessions.
    """

    # Import neccesary packages
    import pandas as pd

    # List all of our datasets in the folder
    datasets = [ds for ds in datasets_path if '.csv' in ds]

    # Load all the datasets in and append it to a list called 'frames'. We can
    # use this list to concatenate all the dataframes according to the Pandas
    # documentation.
    frames = []
    for dataset in datasets:
        df = pd.read_csv(dataset, error_bad_lines=False, index_col=False)
        df = df[df.columns[-5:]]
        df.columns = ['Title', 'Topic', 'Date', 'Content', 'Link']
        frames.append(df)

    # Return a concatenated dataframe of all the dataframes.
    return pd.concat(frames)


#------------------------------------------------#
#                NLP PREPROCESSING               #
#------------------------------------------------#

def remove_clutter(text):

    """
    This function takes a string and removes what we consider as clutter
    in our data. This includes things such as special unicode characters
    and video timestamps.
    """

    # Importing neccesary packages
    import re

    # Trying to remove special unicode characters. Our regular expression finds
    # any substrings that starts with a \x followed by two char/num combinations
    text = re.sub(r'\\x[A-Za-z0-9_]{2}', '', text)

    # Trying to remove video annotation. We are using a regular expression,
    # to find any pattern that matches the word video followed by a
    # timestamp (length of video). We believe this is just clutter in
    # our data as well.
    text = re.sub(r'VIDEO([0-9]|[0-9]{2}):[0-9]{4}:[0-9]{2}', ' ', text)

    # Trying to remove image references. Whenever an article contains an
    # image, the page returns a string representation of the image as the
    # source "Getty Images". We remove this representation, as it brings
    # no value to the analysis.
    text = text.replace('Getty Images', '')

    # We now remove commas, apostrophes, and double spaces. We introduce
    # double spaces in the line above, however this could mess up our
    # tokenization, so we simply convert any doublespaces to single spaces.
    # We remove apostrophes after n's to normalize contracted words like
    # wasn't, couldn't etc. Some of these words are already normalized
    # since some of these apostrophes already have been remove by the regex
    # unicode decluttering.
    text = re.sub(r',','', text)
    text = re.sub(r"n'",'n', text)
    text = re.sub(r'  ',' ', text)

    # Finally, we return the decluttered text.
    return text

def cleaning(df,column):

    """
    This function takes a dataframe and a column name and cleans the entire
    column for clutter (using remove_clutter function), then pass it through
    spaCy to further clean and tokenize. It returns the original dataframe,
    but the given column is now cleared of clutter and it contains two
    additional columns (Tokens, cleaned_text) as well.
    """

    # Import neccesary packages
    import spacy
    nlp = spacy.load('en_core_web_sm')
    import pandas as pd

    # First we instantiate a list that we can append all processed tokens in.
    # This makes it possible for us to append it to the dataframe at a later
    # stage.
    tokens = []

    # Now, we apply our remove_clutter function to the chosen column in the
    # dataframe. This runs the remove_clutter function for every entry in
    # the column.
    df[column].apply(remove_clutter)

    # Define an variable to count the progress of our cleaning.
    index = 0

    # Now, we iterate over all entries (articles in our case) in the column
    # and create a nlp object for each, which we can work with.
    for article in nlp.pipe(df[column], disable=['parser']):

        # Now, we store all tokens that pass our requirements in a list for each
        # article. That means that each article will have their own
        # list of tokens.
        article_tok = [token.lemma_.lower() for token in article if _
            token.is_alpha _
            and not token.is_stop _
            and token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'ADV', 'VERB'] _
            and token.ent_type_ not in ['PERSON', 'MONEY', 'PERCENT', 'LOC', 'DATE', 'TIME', 'QUANTITY', 'ORDINAL'] _
            and len(token)>1]

        # Now, we append said list of tokens for each article in our tokens list.
        tokens.append(article_tok)

        # When each article is processed, we increase the index by one and print
        # the progress. This allows us to keep track of how far it is in the
        # cleaning process. When you are dealing with many thousands of
        # articles, it might take a while, so this feature is quite nice.
        index += 1
        print(f'Processed {index}/{len(df[column])}')

    # When all cleaned articles are appended to our tokens list, we simply
    # add the list as a column in the original dataframe.
    df['tokens'] = tokens

    # Lastly, we reconstruct all the articles from the tokens, simply by joining
    # all the tokens in each article_tok list. We achieve this by a simple
    # combination of map & lambda functions.
    df['clean_articles'] = df['tokens'].map(lambda row: " ".join(row))

    # Returning the df that contains cleaned data and new columns.
    return df


def filter_articles_by_category(article_df, category_map_df):

    """
    This function takes a dataframe with a column of topics, and attempts to
    remap the categories through a mapping dataframe. The mapping dataframe
    consists of two columns: The first column is a list of the unique topics
    in the original dataframe and the second column contains which each topic
    should be remapped to. "Investment with Cramer" could eg. be remapped to
    a more general topic as "Investing".
    """

    # instantiate an empty list
    predetermined = []

    # For all rows in article_df, check whether the topic is mapped in
    # category_map_df
    for topic in article_df["Topic"]:

        # If the topic is in category_map_df, then take the remapped topic and
        # store it in 'predetermined'-list
        if topic in list(category_map_df["Topic"]):
            predetermined.append(category_map_df[category_map_df["Topic"] == topic][1].to_numpy()[0])

        # If it's not in the list, append "Other" to 'predetermined'-list
        else:
            predetermined.append("Other")

    # Now, add predetermined-list to article_df as 'final_topic' and return the
    # dataframe. The final_topic column will contain the remapped categories.
    article_df['final_topic'] = predetermined
    return article_df


#------------------------------------------------#
#                DATA PREPARATION                #
#------------------------------------------------#


def calculate_returns(prices, interval):

    """
    This function takes a dataset with prices for different securities over
    time, where each row is a point in time and each column is a security.
    It then calculates the returns for each security for a given interval
    for at each date. It returns a dataset with dates as rows and securities as
    columns, with returns for the given interval as values.
    """

    # Importing neccesary packages
    import pandas as pd

    # Converting all date-strings in date column to actual date objects. We can
    # use these at a later stage to match returns to news articles.
    prices['Dates'] = pd.to_datetime(prices['Dates']).dt.date

    # Now we instantiate a new list to store our returns in.
    date_index = []

    # For every entry in the prices dataframe, try to fetch the current prices
    # and the prices 'interval' periods in the future. If successful, get the
    # return and append it to a list called 'returns'
    for i in range(0,len(prices)):
        try:
            # Getting the current date of the entry
            date = prices.iloc[i,0]

            # Getting the prices for said date
            prices_at_date = prices.iloc[i,1:]

            # Getting the prices 'interval' periods in the future
            prices_at_future_date = prices.iloc[i+interval,1:]

            # Attempt to calculate the returns between the two periods.
            return_at_date = list(prices_at_future_date / prices_at_date)

            # Create a list called returns that contains the date. We can then
            # append the returns in this list as well.
            returns = [date]
            for sector in return_at_date:
                # For every column (sector) in our returns data, append it to
                # the returns list.
                returns.append(sector)

            # Now, we can take the returns for each date and append it to our
            # date_index list, which will make up our final dataframe in the end.
            date_index.append(returns)
        except:
            # If we can't calculate the returns, simply pass the date.
            pass

    # Now, convert date_index to a dataframe and return the dataframe.
    df = pd.DataFrame(date_index, columns = prices.columns)
    return df

def rolling_articles(start_date, end_date, df, start_range, end_range):

    """
    Just like the rolling returns, this function concatenates all the articles
    into a dataframe consisting of dates as rows. Intially, we played with the
    thought of concatenating the articles rolling with a weeks lag, which is
    why it supprts end and start range, however this demanded too much
    compututional power for our time scope, which is why we simply used
    concatenated them daily.
    """

    # Importing neccesary packages
    from datetime import timedelta, datetime
    import ast

    # Generating a list of dates that we can use to filter the articles from.
    date_list = [start_date + timedelta(days=x) for x in range(0,int((end_date - start_date).days)+1)]

    # Converting all date-strings in date column to actual date objects
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    # Generate new dataframe and instantiating a count variable that we can use
    # to display the progress whilst running it.
    date_index = []
    count = 0

    # For every date in our generated list of dates, find all the articles that
    # lies within the range. Then take their tokens (because of re-import,
    # these were actually a string) and convert to a string objects without
    # list characters. Also, if any date has more than 30 articles,
    # just take the 30 first articles. We integrated the last condition because
    # of diminishing marginal benefit compared to the extra compututional effort.
    for date in date_list:

        # Here we get the articles
        articles = df[(df['Date'] <= date + timedelta(days=end_range)) & (df['Date'] >= date + timedelta(days=start_range))].head(30*(1+end_range-start_range))
        count += len(articles)
        processed = ""
        for article in articles['tokens']:
            try:
                # Here we attempt to remove the list characters. I didn't matter
                # if we passed our articles as tokens or strings to gensim's
                # Word2Vec algo, so we chose as string for the ease of it.
                article = str(article).replace("[", "").replace("]", "").replace(",", "").replace("'","")
                processed += article
                processed += " "
            except:
                pass
        # Now, append the date and the related articles to our date_index list,
        # which we can turn into a dataframe, once it is returned.
        date_index.append([date, processed])
        print(f'{date}: {count}')

    # Finally, return the date_index
    return date_index

#------------------------------------------------#
#                MODEL VALIDATION                #
#------------------------------------------------#

def walk_forward_validation(model, epochs, x, y, step_size, train_steps, val_window):

    """
    This function takes a model, specifically our neural net with multiple
    LSTM-layers, the desired number of epochs, x data, y data, desired step
    size, desired number of steps that should be trained per round, and the
    desired validation window. It then trains a model through a Walk Forward
    Validation method that stores MSE-scores for both training and validation
    steps over time. It returns our backtesting predicted y, our trained y's
    and our MSE-scores.
    """

    # First, we import the required packages. We only have one dependency which
    # we uses to calculate the mean squared error each period
    from sklearn.metrics import mean_squared_error

    # Now we instantiate a couple of things. First we define how many records
    # that we have - We use this to loop through our data. Then we define the
    # initial training size, which gives us the point in time where we
    # should start over test. Then we instantiate three empty lists that we
    # later will use to store our results.
    n_records = len(x)
    n_init_train = step_size * train_steps
    train_pred = []
    val_pred = []
    mse_scores = []

    # This for loop goes from the starting point in time (as defined above)
    # to the end of our data and step through the data, enabling us to make the
    # walk forward validation. Our current point in time, i, will jump by the
    # step size each iteration.
    for i in range(n_init_train, n_records, step_size):

        # We know that the starting point for the training data, must be the
        # current point in time minus the training period.
        train_from = i-n_init_train

        # We need to train it to the current point in time.
        train_to = i

        # We then need to validate starting from tomorrow relative to
        # the current point in time.
        test_from = i+1
        # And validate the desired window in the future relative to the
        # point in time
        test_to = i+val_window

        # Now we can split our data at this point in time
        x_train, x_test = x[train_from:train_to], x[test_from:test_to]
        y_train, y_test = y[train_from:train_to], y[test_from:test_to]

        # And then use the data to train the model
        print(f'Train from {i-n_init_train} to {i} and validate for {i+1} to {i+val_window}')
        model.fit(x_train, y_train, epochs=epochs, verbose=1)

        # Here, we can store the training phase's historical predictions of seen y.
        y_train_pred = model.predict(x_train)
        for y_train_day in y_train_pred:
            train_pred.append(y_train_day.tolist())

        # Here, we store the validation phase's future predictions of unseen y.
        y_pred = model.predict(x_test)
        for y_test_day in y_pred:
            val_pred.append(y_test_day.tolist())

        # Here, we calculate MSE for both and append it to our MSE-scores list.
        train_mse = mean_squared_error(y_train,y_train_pred)
        val_mse = mean_squared_error(y_test,y_pred)
        mse_scores.append([train_mse, val_mse])

        print(f'     train: {train_mse} \nvalidation: {val_mse} \n')

    # Lastly, we return the training predictions, the actual validation
    # predictions as well as the observed MSE-scores.
    return train_pred, val_pred, mse_scores
