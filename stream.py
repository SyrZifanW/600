import sys
import twitter
import sqlite3
import time

def oauth_login():
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    OAUTH_TOKEN = ''
    OAUTH_TOKEN_SECRET = ''

    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)

    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


def twitter_search(**kw):
    # See https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets
    # and https://developer.twitter.com/en/docs/tweets/search/guides/standard-operators
    # for details on advanced search criteria that may be useful for
    # keyword arguments

    twitter_api = oauth_login()

    q = "CrossFit"

    # See https://dev.twitter.com/docs/api/1.1/get/search/tweets
    search_results = twitter_api.search.tweets(q=q, count=100, **kw)

    statuses = search_results['statuses']

    # Iterate through batches of results by following the cursor until we
    # reach the desired number of results, keeping in mind that OAuth users
    # can "only" make 180 search queries per 15-minute interval. See
    # https://developer.twitter.com/en/docs/basics/rate-limits
    # for details. A reasonable number of results is ~1000, although
    # that number of results may not exist for all queries.

    # Enforce a reasonable limit
    max_results = 1000

    for _ in range(10):  # 10*100 = 1000
        try:
            next_results = search_results['search_metadata']['next_results']
        except KeyError as e:  # No more results when next_results doesn't exist
            break

        # Create a dictionary from next_results, which has the following form:
        # ?max_id=313519052523986943&q=NCAA&include_entities=1
        kwargs = dict([kv.split('=')
                       for kv in next_results[1:].split("&")])

        search_results = twitter_api.search.tweets(**kwargs)
        statuses += search_results['statuses']

        if len(statuses) > max_results:
            break

    return statuses


def get_stream():
    cycle = 0
    usa = {"USA", 'US', 'U.S', 'U.S.A', 'UNITED STATES OF AMERICA', 'THE UNITED STATES OF AMERICA', 'THE UNITED STATES'}
    states = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'D.C.', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
              'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'}
    full_states = {'ALABAMA': 'AL', 'INDIANA': 'ID', 'NEBRASKA': 'NE', 'SOUTH CAROLINA': 'SC', 'ALASKA': 'AK',
                   'IOWA': 'IA', 'NEVADA': 'NV', 'SOUTH DAKOTA': 'SD', 'ARIZONA': 'AZ', 'KANSAS': 'KS',
                   'NEW HAMPSHIRE': 'NH', 'TENNESSEE': 'TN', 'ARKANSAS': 'AR', 'KENTUCKY': 'KY', 'NEW JERSEY': 'NJ',
                   'TEXAS': 'TX', 'CALIFORNIA': 'CA', 'LOUISIANA': 'LA', 'NEW MEXICO': 'NM',
                   'UTAH': 'UT', 'COLORADO': 'CO', 'MAINE': 'ME', 'NEW YORK': 'NY', 'VERMONT': 'VT',
                   'CONNECTICUT': 'CT', 'MARYLAND': 'MD', 'NORTH CAROLINA': 'NC', 'VIRGINIA': 'VA',
                   'DELAWARE': 'DE', 'MASSACHUSETTS': 'MA', 'NORTH DAKOTA': 'ND', 'WASHINGTON': 'WA', 'FLORIDA': 'FL',
                   'MICHIGAN': 'MI', 'OHIO': 'OH', 'WEST VIRGINIA': 'WV', 'GEORGIA': 'GA',
                   'MINNESOTA': 'MN', 'OKLAHOMA': 'OK', 'WISCONSIN': 'WI', 'HAWAII': 'HI', 'MISSISSIPPI': 'MS',
                   'OREGON': 'OR', 'WYOMING': 'WY', 'IDAHO': 'ID', 'MISSOURI': 'MO',
                   'PENNSYLVANIA': 'PA', 'ILLINOIS': 'IL', 'MONTANA': 'MT', 'RHODE ISLAND': 'RI'}
    cities = {'NEW YORK CITY':'NY','NYC':'NY','NEW YORK':'NY', 'LOS ANGELES':'CA', 'CHICAGO':'IL', 'HOUSTON':'TX', 'PHOENIX':'AZ', 'PHILADELPHIA':'PA', 'SAN ANTONIO':'TX',
           'SAN JOSE':'CA', 'AUSTIN':'TX', 'JACKSONVILLE':'FL', 'FORT WORTH':'TX', 'COLUMBUS':'OH', 'DAN FRANCISCO':'CA', 'CHARLOTTE':'NC', 'INDIANAPOLIS':'IN', 'SEATTLE':'WA',
           'DENVER':'CO', 'WASHINGTON':'D.C.', 'BOSTON':'MA', 'EI PASO':'TX', 'DETROIT':'MI', 'NASHVILLE':'TN', 'PORTLAND':'OR', 'OKLAHOMA CITY':'TN', 'LAS VEGAS':'NV',
           'LOUISVILLE':'KY', 'BALTIMORE':'MD', 'MILWAUKEE':'WI', 'ALBUQUERQUE':'NM', 'TUCSON':'AZ', 'FRESNO':'CA', 'MESA':'AZ', 'SACRAMENTO':'CA', 'ATLANTA':'GA', 'KANSAS CITY':'KS',
           'COLORADO SPRINGS':'CO', 'MIAMI':'FL', 'RALEIGH':'NC', 'OMAHA':'NE', 'LONG BEACH':'CA', 'VIRGINIA BENCH':'VA', 'OAKLAND':'CA', 'MINNEAPOLIS':'MN', 'TULSA':'OK',
           'ARLINGTON':'TX', 'TAMPA':'FL', 'NEW ORLEANS':'LA', 'WICHITA':'KS', 'CLEVELAND':'OH', 'BAKERSFIELD':'CA', 'AURORA':'CO', 'ANAHEIM':'CA', 'HONOLULU':'HI', 'SANTA ANA':'CA',
           'RIVERSIDE':'CA', 'CORPUS CHRISTI':'TX', 'LEXINGTON':'KY', 'STOCKTON':'CA', 'HENDERSON':'NV', 'SAINT PAUL':'MN', 'ST. LOUIS':'MO', 'CINCINNATI':'OH', 'PITTSBURGH':'PA',
           'GREENSBORO':'NC', 'ANCHORAGE':'AK', 'PLANO':'TX', 'LINCOLN':'NE', 'ORLANDO':'FL', 'IRVINE':'CA', 'NEWARK':'NJ', 'TOLEDO':'OH', 'DURHAM':'NC', 'CHULA VISTA':'CA',
           'FORT WAYNE':'IN', 'JERSEY CITY':'NJ', 'ST. PETERSBURG':'FL', 'LAREDO':'TX', 'MADISON':'WI', 'CHANDLER':'AZ', 'BUFFALO':'NY', 'LUBBOCK':'TX', 'SCOTTSDALE':'AZ', 'RENO':'NV',
           'GLENDALE':'AZ', 'BILBERT':'AZ', 'WINSTON-SALEM':'NC', 'NORTH LAS VEGAS':'NV', 'NORFOLK':'VA', 'CHESAPEAKE':'VA', 'GARLAND':'TX', 'IRVING':'TX', 'HIALEAH':'FL', 'FREMONT':'CA',
           'BOISE':'ID','RICHMOND':'VA', 'BATON ROUGE':'LA', 'SPOKANE':'WA', 'DES MOINES':'IA', 'TACOMA':'WA', 'SAN BERNARDINO':'CA', 'MODESTO':'CA', 'FONTANA':'CA', 'SANTA CLARITA':'CA',
           'BIRMINGHAM':'AL', 'OXNARD':'CA', 'FAYETTEVILLE':'NC', 'MORENO VALLEY':'CA', 'ROCHESTER':'NY', 'HUNTINGTON BEACH':'CA', 'SALT LAKE CITY':'UT', 'GRAND RAPIDS':'MI',
           'AMARILLO':'TX', 'YONKERS':'NY', 'MONTGOMERY':'ALABAMA', 'AKRON':'OH', 'LITTLE ROCK':'AR', 'HUNTSVILLE':'AL', 'AUGUSTA':'GA', 'PORT ST.LUCIE':'FL', 'GRAND PRAIRIE':'TX',
           'TALLAHASSEE':'FL', 'OVERLAND PARK':'KS', 'TEMPE':'AZ', 'MCKINNEY':'TX', 'MOBILE':'AL', 'CAPE CORAL':'FL', 'SHREVEPORT':'LA', 'FRISCO':'TX', 'KNOXVILLE':'TN', 'WORCESTER':'MA',
           'BROWNSVILLE':'TX', 'VANCOUVER':'WA', 'FORT LAUDERDALE':'FL', 'SIOUX FALLS':'SD', 'ONTARIO':'CA', 'CHATTANOOGA':'TN', 'PROVIDENCE':'RI', 'NEWPORT NEWS':'VA',
           'RANCHO CUCAMONGA':'CA', 'SANTA ROSA':'CA', 'OCEANSIDE':'CA', 'SALEM':'OR', 'ELK GROVE':'CA', 'GARDEN GROVE':'CA', 'PEMBROKE PINES':'FL', 'PEORIA':'AZ', 'EUGENE':'OR',
           'CORONA':'CA', 'CARY':'NC', 'SPRINGFIELD':'MO', 'FORT COLLINS':'CO', 'JACKSON':'MS', 'ALEXANDRIA':'VA', 'HAYWARD':'CA', 'LANCASTER':'CA', 'LAKEWOOD':'CO', 'CLARKSVILLE':'TN',
           'PALMDALE':'CA', 'SALINAS':'CA', 'HOLLYWOOD':'FL', 'PASADENA':'TX', 'SUNNYVALE':'CA', 'MACON':'GA', 'POMONA':'CA', 'ESCONDIDO':'CA', 'KILEEN':'TX', 'NAPERVILLE':'IL',
           'JOLIET':'IL', 'BELLEVUE':'WA', 'ROCKFORD':'IL', 'SAVANNAH':'GA', 'PATERSON':'NJ', 'TORRANCE':'CA', 'BRIDGEPORT':'CT', 'MCALLEN':'TX', 'MESQUITE':'TX', 'SYRACUSE':'NY',
           'MIDLAND':'TX', 'MURFREESBORO':'TN', 'MIRAMAR':'FL', 'DAYTON':'OH', 'FULLERTON':'CA', 'OLATHE':'KS', 'ORANGE':'CA', 'THORNTON':'CO', 'ROSEVILLE':'CA', 'DENTON':'TX',
           'WACO':'TX', 'SURPRISE':'AZ', 'CARROLTON':'TX', 'WEST VALLEY CITY':'UT', 'CHARLESTON':'SC', 'WARREN':'MI', 'HAMPTON':'VA', 'GAINESVILLE':'FL', 'VISALIA':'CA',
           'CORAL SPRINGS':'FL', 'COLUMBIA':'SC', 'CEDAR RAPIDS':'IA', 'STERLING HEIGHTS':'MI', 'NEW HAVEN': 'CT', 'STAMFORD':'CT', 'CONCORD':'CA', 'KENT':'WA', 'SANTA CLARA':'CA',
           'ELIZABETH':'NJ', 'ROUND ROCK':'TX', 'THOUSAND OAKS':'CA', 'LAFAYETTE':'LA', 'ATHENS':'GA', 'TOPEKA':'KS', 'SIMI VALLEY':'CA', 'FARGO':'ND', 'NORMAN':'OK', 'ABILENE':'TX',
           'WILMINGTON':'NC', 'HARTFORD':'CT', 'VICTORVILLE':'CA', 'PEARLAND':'TX', 'VALLEJO':'CA', 'ANN ARBOR':'MI', 'BERKELEY':'CA', 'ALLENTOWN':'PA', 'RICHARDSON':'TX',
           'ODESSA':'TX', 'ARVADA':'CO', 'CAMBRIDGE':'MA', 'SUGAR LAND':'TX', 'BEAUMONT':'TX', 'LANSING':'MI', 'EVANSVILLE':'IN', 'INDEPENDENCE':'MO', 'FAIRFIELD':'CA', 'PROVO':'UT',
           'CLEARWATER':'FL', 'COLLEGE STATION':'TX', 'WEST JORDAN':'UT', 'CARLSBAD':'CA', 'EL MONTE':'CA', 'MURRIETA':'CA', 'TEMECULA':'CA', 'PALM BAY':'FL', 'WESTMINSTER':'CO',
           'NORTH CHARLESTON':'SC', 'MIAMI GARDENS':'FL', 'MANCHESTER':'NH', 'HIGH POINT':'NC', 'DOWNEY':'CA', 'CLOVIS':'CA', 'POMPANO BEACH':'FL', 'PUEBLO':'CO', 'ELGIN':'IL',
           'LOWELL':'MA', 'ANTIOCH':'CA', 'WEST PALM BEACH':'FL', 'EVERETT':'WA', 'VENTURA':'CA', 'CENTENNIAL':'CO', 'LAKELAND':'FL', 'GRESHAM':'OR', 'BILLINGS':'MT', 'INGLEWOOD':'CA',
           'BROKEN ARROW':'OK', 'SANDY SPRINGS':'GA', 'JURUPA VALLEY':'CA', 'HILLSBORO':'OR', 'WATERBURY':'CT', 'SANTA MARIA':'CA', 'BOULDER':'CO', 'GREELEY':'CO', 'DALY CITY':'CA',
           'MERIDIAN':'ID', 'LEWISVILLE':'TX', 'DAVIE':'FL', 'WEST COVINA':'CA', 'LEAGUE CITY':'TX', 'TYLER':'TX', 'NORWALK':'CA', 'SAN MATEO':'CA', 'GREEN BAY':'WI', 'WICHITA FALLS':'TX',
           'SPARKS':'NV', 'BURBANK':'CA', 'RIALTO':'CA', 'ALLEN':'TX', 'EL CAJON':'CA', 'LAS CRUCES':'NM', 'RENTON':'WA', 'DAVENPORT':'IA', 'VISTA':'CA', 'TUSCALOOSA':'AL', 'CLINTON':'MI',
           'EDISON':'NJ', 'WOODBRIDGE':'NJ', 'SAN ANGELO':'TX', 'KENOSHA':' WI', 'VACAVILLE':'CA','SAN DIEGO':'CA', 'DALLAS':'TX',}

    while True:
        cycle += 1
        db = sqlite3.connect("tweets.db")
        cursor = db.cursor()
        # Query terms
        q = 'corona,coronavirus,COVID-19' # Comma-separated list of terms

        twitter_api = oauth_login()
        twitter_stream = twitter.TwitterStream(auth=twitter_api.auth)
        stream = twitter_stream.statuses.filter(track=q, language='en')

        count = 1
        try:
            for tweet in stream:
                if count % 200 == 0:
                    print(f"Cycle {cycle}: Committed {count}")
                    db.commit()
                if 'id' not in tweet :
                    continue
                else:
                    id = tweet['id']
                # if 'lang' in tweet and tweet['lang'] != 'en':
                #     print(tweet['text'])
                #     continue
                if 'extended_tweet' in tweet:
                    text = tweet['extended_tweet']['full_text']
                elif 'text' in tweet:
                    text = tweet['text']
                    if text.endswith('…'):
                        continue
                else:
                    continue
                if 'created_at' in tweet:
                    created_time = tweet['created_at'] # str
                else:
                    created_time = ''
                if 'in_reply_to_status_id_str' in tweet:
                    in_reply_to_status_id_str = tweet['in_reply_to_status_id_str']
                else:
                    in_reply_to_status_id_str = ''
                if 'in_reply_to_user_id_str' in tweet:
                    in_reply_to_user_id_str = tweet['in_reply_to_user_id_str']
                else:
                    in_reply_to_user_id_str = ''
                if 'user' in tweet:
                    user_info = tweet['user']
                    if 'name' in user_info:
                        user_name = user_info['name']
                    else:
                        user_name = ''
                    if 'screen_name' in user_info:
                        user_screen_name = user_info['screen_name']
                    else:
                        user_screen_name = ''
                    if 'id_str' in user_info:
                        user_id_str = user_info['id_str']
                    else:
                        user_id_str = ''
                    if 'location' in user_info:
                        user_location = user_info['location']
                        if user_location:
                            user_location = user_location.strip()
                            if "," in user_location:
                                locations = user_location.split(",")
                                if len(locations) == 2 or len(locations) == 3:
                                    l = locations[1].strip().upper()
                                    if l in usa:
                                        # city, usa
                                        city = locations[0].strip().upper()
                                        if city in cities:
                                            state_location = cities[city]
                                        elif city in states:
                                            state_location = city
                                        elif city in full_states:
                                            state_location = full_states[city]
                                        else:
                                            state_location = 'USA'
                                    elif l in states:
                                        # city, state 或者 city, state, usa
                                        state_location = l
                                    elif l in full_states:
                                        # city, state 或者 city, state, usa
                                        state_location = full_states[l]
                                    elif l in cities:
                                        # xxx, city, state/usa
                                        state_location = cities[l]
                                    else:
                                        state_location = ''
                                else:
                                    # 三个逗号以上的认为是无效地址
                                    state_location = ''
                            else:
                                # 无分隔， 可能是 city 或者 state 或者 usa
                                l = user_location.upper()
                                if l in cities:
                                    state_location = cities[l]
                                elif l in states:
                                    state_location = l
                                elif l in full_states:
                                    state_location = full_states[l]
                                elif l in usa:
                                    state_location = 'USA'
                                else:
                                    state_location = ''
                        else:
                            state_location = ''
                    else:
                        user_location = ''
                        state_location = ''
                    if 'followers_count' in user_info:
                        user_followers_count = user_info['followers_count']
                    else:
                        user_followers_count = -1
                    if 'friends_count' in user_info:
                        user_friends_count = user_info['friends_count'] # following count
                    else:
                        user_friends_count = -1
                    if 'created_at' in user_info:
                        user_created_time = user_info['created_at']
                    else:
                        user_created_time = ''
                else:
                    continue
                if 'entities' in tweet:
                    hash_tag = [dic['text'] for dic in tweet['entities']['hashtags']]
                else:
                    hash_tag = ''
                try:
                    cursor.execute("insert into stream (id, created_time, content, hash_tags, name, screen_name, user_id, user_location, followers_count, friends_count, user_created_time, tweet_id, reply_to_tweet_id, reply_to_user_id, state) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                   (count, created_time, text, '#'.join(hash_tag), user_name, user_screen_name, user_id_str, user_location, user_followers_count, user_friends_count,
                                    user_created_time, id, in_reply_to_status_id_str, in_reply_to_user_id_str, state_location))
                except sqlite3.IntegrityError:
                    print("exist")

                count += 1
                sys.stderr.flush()
        finally:
            print("error")
            db.commit()
            db.close()
        print("Sleeping...")
        time.sleep(3600*10)

get_stream()
