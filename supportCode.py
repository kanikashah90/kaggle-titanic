total_input_size = np.size(data[0:, 0])

# See the correlation between Survived and Sex.
female_mask = data[0:, 4] == "female"
male_mask = data[0:, 4] != "female"

women_onboard = data[female_mask, 1].astype(np.float)
men_onboard = data[male_mask, 1].astype(np.float)
perc_women_survived = np.sum(women_onboard)/np.size(women_onboard)
perc_men_survived = np.sum(men_onboard)/np.size(men_onboard)

#print 'women survival percentage is %f ' % (perc_women_survived)
#print 'men survival percentage is %f ' % (perc_men_survived)
# Women are more likely to survive then men. Almost 4 times more.

# See if there is a correlation between Survived and PClass
passenger_class_survived = data[data[0:, 1].astype(np.int32) == 1, 2].astype(np.int32)
passenger_class_died = data[data[0:, 1].astype(np.int32) == 0, 2].astype(np.int32)
print 'survival percentage in classes: ', (np.bincount(passenger_class_survived).astype(np.float) / np.size(data[0:, 0]))
print 'not survival percentage in classes: ', (np.bincount(passenger_class_died).astype(np.float) / np.size(data[0:, 0]))
# Not very predictive, but it seems like class 1 has higher survival percentage
# and class 3 has a very high dying percentage.
# Let's see if gender has any impact on the class.
# So now there are two dependent discrete variables: gender and class
women_class_survived = data[(data[0:, 4] == "female") & (data[0:, 1].astype(np.int32) == 1), 2].astype(np.int32)
women_class_died = data[(data[0:, 4] == "female") & (data[0:, 1].astype(np.int32) == 0), 2].astype(np.int32)
classes_total_across_women = data[data[0:, 4] == "female", 2].astype(np.int32)
classes_total_across_women = np.bincount(classes_total_across_women)
print '\n 0. Filler class 1: Class 1 2: Class 2 3: Class 3'
print 'women survival percentage across classes:', np.bincount(women_class_survived).astype(np.float) / classes_total_across_women
print 'women not survival percentage across classes:', np.bincount(women_class_died).astype(np.float) / classes_total_across_women
# Class 1 women have higher probability of surviving. However no other class women survivale chances can be assured upfront.

# See if Age has any influence on the survival
# Distribution is almost identical in this case. Need to break it down
# Split age in two groups. age < 15 and age > 15
passenger_with_age_mask = data[0:, 5] != ''
passengers_age = data[passenger_with_age_mask, 5].astype(np.float)
passengers_age[passengers_age <= 15] = 0
passengers_age[passengers_age > 15] = 1
passengers_age = passengers_age.astype(int)
passenger_age_count = np.bincount(passengers_age)
passenger_age_revealed_survived = data[(passenger_with_age_mask) & (data[0:, 1].astype(np.int32) == 1), 5].astype(np.float)
passenger_age_revealed_died = data[(passenger_with_age_mask) & (data[0:, 1].astype(np.int32) == 0), 5].astype(np.float)
passenger_age_revealed_survived[passenger_age_revealed_survived <= 15] = 0 # 0 is for Kids
passenger_age_revealed_survived[passenger_age_revealed_survived > 15] = 1 # 1 is for Adult

passenger_age_revealed_died[passenger_age_revealed_died <= 15] = 0 # 0 is for Kids
passenger_age_revealed_died[passenger_age_revealed_died > 15] = 1 # 1 is for Adult
passenger_age_revealed_survived = passenger_age_revealed_survived.astype(int)
passenger_age_revealed_died = passenger_age_revealed_died.astype(int)

print '\n0: Kids, 1: Adults'
print 'survival percentage based on age:', (np.bincount(passenger_age_revealed_survived).astype(np.float)/ passenger_age_count)
print 'death percentage based on age:', (np.bincount(passenger_age_revealed_died).astype(np.float)/ passenger_age_count)
# Age doesn't seem like a possible candidate to contribute towards the survival.

# Relationship between Ticket and Fare
# Some tickets are alphanumeric while some are just numbers.
# Get the fare range of alphanumeric tickets
passenger_ticket = data[0:, 8]
# Write a mask which can check for alphanumeric ticket numbers.
ticket_alnum_mask = np.array([bool(re.search('[a-zA-Z]', element)) for element in data[0:, 8]])
ticket_alnum_mask = ticket_alnum_mask.astype(int)
ticket_category_count = np.bincount(ticket_alnum_mask)
# Create two kind of category for tickets, tickets with alphanumeric value and tickets
# without alphanumeric value
# There doesn't seem to be readily available relationship between fare and ticket number
# But now I am interested to know the relationship between people with alphanumeric tickets and there survival.
survivors_mask = data[0:,1].astype(np.float) == 1
passenger_ticket_survived = data[data[0:, 1].astype(np.float) == 1, 8]
passenger_ticket_died = data[data[0:, 1].astype(np.float) == 0, 8]
passenger_survived_alnum_mask = np.array([bool(re.search('[a-zA-Z]', element)) for element in passenger_ticket_survived])
passenger_survived_alnum_mask = passenger_survived_alnum_mask.astype(int)
passenger_died_alnum_mask = np.array([bool(re.search('[a-zA-Z]', element)) for element in passenger_ticket_died])
passenger_died_alnum_mask = passenger_died_alnum_mask.astype(int)

print '\n 0: NumericTicket 1: Alphanumeric Ticket'
print 'survival percentage based on ticket:', (np.bincount(passenger_survived_alnum_mask).astype(np.float) / ticket_category_count)
print 'death percentage based on ticket: ', (np.bincount(passenger_died_alnum_mask).astype(np.float) / ticket_category_count)
# There is no relationship between ticket being alphanumeric and person surviving.

# Also, by splitting the fare into plausible classes I will be interested to know if fare has any relationship with
# the person's survival
# Ticket fare is varying from 0 - 512.
# Lets create 2 classes for the fare:
# 1. 0 - 100
# 2. > 100
ticket_fare = data[0:, 9].astype(np.float)
ticket_fare[(ticket_fare >= 0.0) & (ticket_fare < 100)] = 1
ticket_fare[ticket_fare >= 100] = 2
ticket_fare = ticket_fare.astype(int)
ticket_fare_count = np.bincount(ticket_fare)
passenger_fare_survived = data[data[0:, 1].astype(np.float) == 1, 9].astype(np.float)
passenger_fare_died = data[data[0:, 1].astype(np.float) == 0, 9].astype(np.float)
passenger_fare_survived[(passenger_fare_survived >= 0.0) & (passenger_fare_survived < 100)] = 1
passenger_fare_survived[(passenger_fare_survived >= 100)] = 2
passenger_fare_died[(passenger_fare_died >= 0.0) & (passenger_fare_died < 100)] = 1
passenger_fare_died[(passenger_fare_died >= 100)] = 2
passenger_fare_survived = passenger_fare_survived.astype(int)
passenger_fare_died = passenger_fare_died.astype(int)

print '\n 0: 0 <= Fare < 100 1: Fare > 100'
print 'survival percentage based on fare:', (np.bincount(passenger_fare_survived).astype(np.float) / ticket_fare_count)
print 'death percentage based on fare:', (np.bincount(passenger_fare_died).astype(np.float) / ticket_fare_count)
# Fare paid to get on Titanic seems lika a good indicator of survival.
# Anyone boarded with fare < 100 has good chance of survival compared to anybody else.

# See if a SibSp has any influence on the Survived.
passenger_sibsp = data[0:, 6].astype(int)
passenger_sibsp_count = np.bincount(passenger_sibsp)
passenger_sibsp_survived = data[data[0:, 1].astype(np.float) == 1, 6].astype(np.int)
passenger_sibsp_died = data[data[0:, 1].astype(np.float) == 0, 6].astype(np.int)
passenger_sibsp_survived_count = np.bincount(passenger_sibsp_survived).astype(np.float)
passenger_sibsp_died_count = np.bincount(passenger_sibsp_died).astype(np.float)
print '\n'
print 'survival percentage based on sipsp:', (passenger_sibsp_survived_count / passenger_sibsp_count[0:len(passenger_sibsp_survived_count)])
print 'death percentage based on sipsp:', (passenger_sibsp_died_count / passenger_sibsp_count[0: len(passenger_sibsp_died_count)])
# High value of sbsp means less chances of survival

# Does the Parch plays any role.
passenger_parch = data[0:, 7].astype(int)
passenger_parch_count = np.bincount(passenger_parch)
passenger_parch_survived = data[data[0:, 1].astype(np.float) == 1, 7].astype(np.int)
passenger_parch_died = data[data[0:, 1].astype(np.float) == 0, 7].astype(np.int)
passenger_parch_survived_count = np.bincount(passenger_parch_survived).astype(np.float)
passenger_parch_died_count = np.bincount(passenger_parch_died).astype(np.float)
print '\n'
print 'survival percentage based on parch:', (passenger_parch_survived_count / passenger_parch_count[0: len(passenger_parch_survived_count)])
print 'death percentage based on parch:', (passenger_parch_died_count / passenger_parch_count[0: len(passenger_parch_died_count)])
# it is kind of bell curve: lover parch seems low survival same as large value for parch

# How is Embarked related to the survival of the passenger.
# classify the value of embarked feature in these categories:
# Value Q -- Cat 1
# Value S -- Cat 2
# Value C -- Cat 3
# Any other value apart from empty -- Cat 4
passenger_embarked_mask = data[0:, 11] != ''
passenger_embarked = data[passenger_embarked_mask, 11]
passenger_embarked[(passenger_embarked != 'Q') &  (passenger_embarked != 'S') & (passenger_embarked != 'C')] = '4'
passenger_embarked[passenger_embarked == 'Q'] = '1'
passenger_embarked[passenger_embarked == 'S'] = '2'
passenger_embarked[passenger_embarked == 'C'] = '3'
passenger_embarked = passenger_embarked.astype(int)
passenger_embarked_count = np.bincount(passenger_embarked)
passenger_embarked_survived = data[((data[0:, 1].astype(int) == 1) & (passenger_embarked_mask)), 11]
passenger_embarked_died = data[((data[0:, 1].astype(int) == 0) & (passenger_embarked_mask)), 11]
passenger_embarked_survived[(passenger_embarked_survived != 'Q') &  (passenger_embarked_survived != 'S') & (passenger_embarked_survived != 'C')] = '4'
passenger_embarked_survived[passenger_embarked_survived == 'Q'] = '1'
passenger_embarked_survived[passenger_embarked_survived == 'S'] = '2'
passenger_embarked_survived[passenger_embarked_survived == 'C'] = '3'
passenger_embarked_survived = passenger_embarked_survived.astype(int)
passenger_embarked_survived_count = np.bincount(passenger_embarked_survived).astype(float)

passenger_embarked_died[(passenger_embarked_died != 'Q') &  (passenger_embarked_died != 'S') & (passenger_embarked_died != 'C')] = '4'
passenger_embarked_died[passenger_embarked_died == 'Q'] = '1'
passenger_embarked_died[passenger_embarked_died == 'S'] = '2'
passenger_embarked_died[passenger_embarked_died == 'C'] = '3'
passenger_embarked_died = passenger_embarked_died.astype(int)
passenger_embarked_died_count = np.bincount(passenger_embarked_died).astype(float)
print '\n 0: Filler Port 1: Port Q 2: Port S 3: Port C 4: Any other port'
print 'survival percentage based on embarkment:', (passenger_embarked_survived_count / passenger_embarked_count[0: len(passenger_embarked_survived_count)])
print 'Death percentage based on embarkment:', (passenger_embarked_died_count / passenger_embarked_count[0: len(passenger_embarked_died_count)])
# Embarkment doesn't tell much