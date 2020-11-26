
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

    # We instanciate an empty list that we can store all the links in.
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
                # href (actual links) in our instanciated list from before.
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

    # First, we instanciate an empty list, that we can append the scraped
    # data to. Then we instanciate an index, which enables us to follow the
    # scrapers progress, and lastly, we instanciate a request Session, which
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
                article = soup_link.find('div', class_="ArticleBody-articleBody").get_text()

                # The date generated from the link. The link will always
                # contain the date if it is an article. We save it as
                # DD/MM/YYYY.
                date = f'{link[29:31]}-{link[26:28]}-{link[21:25]}'

                # The link that have a class of 'ArticleHeader-eyebrow',
                # which is their article topic.
                topic = soup_link.find('a', class_="ArticleHeader-eyebrow").get_text()

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

    # Trying to remove special unicode characters. ord() returns the unicode
    # index of the character - We then check that whether that character
    # index if less than 127. If it is so, it is not a special character
    # and we include it.
    text = ''.join([x for x in text if ord(x) < 127])

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

    # First we instanciate a list that we can append all processed tokens in.
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
    for article in nlp.pipe(df[column], disable=["parser"]):

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
    df["clean_articles"] = df["tokens"].map(lambda row: " ".join(row))

    # Returning the df that contains cleaned data and new columns.
    return df
