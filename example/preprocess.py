from setting import *
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import func, and_
import pandas as pd

Base = declarative_base()


# Declare mapping here

class User(Base):
    __tablename__ = 'users'

    uid = Column(Integer, nullable=False)
    name = Column(String(30), primary_key=True)
    joindate = Column(Date, nullable=False)

    def __repr__(self):
        return "<Users(uid='%s', name='%s', joindate='%s')>" % (
            self.uid, self.name, self.joindate)


class Subject(Base):
    __tablename__ = 'subject'

    id = Column(Integer, nullable=False, primary_key=True)
    name = Column(String(100), nullable=False)
    type = Column(String(5), nullable=False)
    date = Column(Date)
    rank = Column(Integer)
    favnum = Column(Integer, nullable=False, default=0)
    votenum = Column(Integer, nullable=False, default=0)

    def __repr__(self):
        return "<Subject(id='%s', name='%s', type='%s')>" % (
            self.id, self.name, self.type
        )


class Record(Base):
    __tablename__ = 'record'

    name = Column(String(100), primary_key=True, nullable=False)
    typ = Column(String(5), nullable=False)
    iid = Column(Integer, primary_key=True, nullable=False)
    state = Column(String(7), nullable=False)
    adddate = Column(Date, nullable=False)
    rate = Column(Integer)
    tags = Column(String(500))

    def __repr__(self):
        return "<Record(name='%s', iid='%s', rate='%s')>" % (
            self.name, self.iid, self.rate)


def run():
    engine = create_engine("mysql+mysqldb://" + MYSQL_USER + ":" + MYSQL_PASSWD + "@" +
                           MYSQL_HOST + "/" + MYSQL_DBNAME +
                           "?charset=utf8&use_unicode=0")
    Base.metadata.create_all(engine)

    # Session is a custom class
    Session = sessionmaker(bind=engine)
    session = Session()

    value = []
    for q in session.query(Record.name, Record.iid, Record.rate).filter(
            and_(Record.rate != None, Record.typ == 'anime')).all():
        value.append([q.name, q.iid, int(q.rate)])
    table = pd.DataFrame(value, columns=['username', 'itemid', 'rate'])
    table.to_hdf("Data/input.bin", key="user_item_rate", mode='w')


###
# Test if valid
def test():
    engine = create_engine("mysql+mysqldb://" + MYSQL_USER + ":" + MYSQL_PASSWD + "@" +
                           MYSQL_HOST + "/" + MYSQL_DBNAME +
                           "?charset=utf8&use_unicode=0")
    Base.metadata.create_all(engine)

    # Session is a custom class
    Session = sessionmaker(bind=engine)
    session = Session()

    table = pd.read_hdf("Data/input.bin", "user_item_rate")
    cnt = table.shape[0]

    from sqlalchemy import and_
    cnt2 = int(session.query(Record).filter(and_(Record.rate != None,
                                                 Record.typ == 'anime')).count())

    assert (cnt == cnt2)


if __name__ == '__main__':
    run()
