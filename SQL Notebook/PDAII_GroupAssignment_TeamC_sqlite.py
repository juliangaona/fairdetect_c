
import pandas as pd
df = pd.read_csv("synthetic_credit_card_approval.csv")
df = df.fillna(0)

# #CREATING A CONNECTION WITH THE DATABASE ENGINE, IT CAN BE ANY SQL Server, MySql, Posgres, etc...
from sqlalchemy import create_engine
engine = create_engine('sqlite:///./cards_final.db')

#/Users/hebamushtaha/IE-MBD-Python for Data Analysis II/Group Assignment/Final


# DECLARING THE OBJECT - Object-relational mapping
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

# EXTENDING THE Base OBJECT INTO AN OBJECT CALLED SyntheticCardsApprove

from sqlalchemy import Column, Integer
class SyntheticCardsApprove(Base):
    __tablename__ = 'cards_final'
    id = Column(Integer, primary_key=True)
    Num_Children= Column(Integer) 
    Group = Column(Integer)
    Income = Column(Integer)
    Own_Car = Column(Integer)
    Own_Housing = Column(Integer)
    Target = Column(Integer)
    
#SyntheticCardsApprove()
# WE RUN create_all SO THAT IF THE TABLE DOES NOT EXIST IT CREATES IT
Base.metadata.create_all(engine)

# WE CREATE A SESSION TO BE ABLE TO INSERT, UPDATE AND DELETE DATA.
from sqlalchemy.orm import sessionmaker
DBSession = sessionmaker(bind=engine)
session = DBSession()

df.columns

for i in range(0,5): #range(len(df)):
    print(i, df['Num_Children'].iloc[i], df['Group'].iloc[i], df['Income'].iloc[i])


# inserting in the db
for i in range(len(df)):
    CardsApproval = SyntheticCardsApprove(
        Num_Children=int(df['Num_Children'].iloc[i]),
        Group=int(df['Group'].iloc[i]),
        Income=int(df['Income'].iloc[i]),
        Own_Car=int(df['Own_Car'].iloc[i]),
        Own_Housing=int(df['Own_Housing'].iloc[i]),
        Target=int(df['Target'].iloc[i]))
                           
    session.add(CardsApproval)


session.commit()
#session.close()

