usa = {"USA", 'US', 'U.S', 'U.S.A', 'UNITED STATES OF AMERICA', 'THE UNITED STATES OF AMERICA', 'THE UNITED STATES'}
full_states = {'ALABAMA':'AL', 'INDIANA':'ID', 'NEBRASKA':'NE', 'SOUTH CAROLINA':'SC', 'ALASKA':'AK', 'IOWA':'IA', 'NEVADA':'NV', 'SOUTH DAKOTA':'SD', 'ARIZONA':'AZ', 'KANSAS':'KS',
               'NEW HAMPSHIRE':'NH', 'TENNESSEE':'TN', 'ARKANSAS':'AR', 'KENTUCKY':'KY', 'NEW JERSEY':'NJ', 'TEXAS':'TX', 'CALIFORNIA':'CA', 'LOUISIANA':'LA', 'NEW MEXICO':'NM',
               'UTAH':'UT', 'COLORADO':'CO', 'MAINE':'ME', 'NEW YORK':'NY', 'VERMONT':'VT', 'CONNECTICUT':'CT', 'MARYLAND':'MD', 'NORTH CAROLINA':'NC', 'VIRGINIA':'VA',
               'DELAWARE':'DE', 'MASSACHUSETTS':'MA', 'NORTH DAKOTA':'ND', 'WASHINGTON':'WA', 'FLORIDA':'FL', 'MICHIGAN':'MI', 'OHIO':'OH', 'WEST VIRGINIA':'WV', 'GEORGIA':'GA',
               'MINNESOTA':'MN', 'OKLAHOMA':'OK', 'WISCONSIN':'WI', 'HAWAII':'HI', 'MISSISSIPPI':'MS', 'OREGON':'OR', 'WYOMING':'WY', 'IDAHO':'ID', 'MISSOURI':'MO',
               'PENNSYLVANIA':'PA', 'ILLINOIS':'IL', 'MONTANA':'MT', 'RHODE ISLAND':'RI'}
states = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'D.C.', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
          'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'}
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

user_location = # 从csv获取的location, 需要自己赋值
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