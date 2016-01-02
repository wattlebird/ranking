from setting import *
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import func, and_
import struct

Base = declarative_base()

# Declare mapping here

class User(Base):
    __tablename__ = 'users'

    uid = Column(Integer,nullable=False)
    name = Column(String(30),primary_key=True)
    joindate = Column(Date,nullable=False)

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
        return "<Subject(id='%s', name='%s', type='%s')>"%(
        self.id, self.name, self.type
        )

class Record(Base):
    __tablename__ = 'record'

    name = Column(String(100),primary_key=True,nullable=False)
    typ = Column(String(5),nullable=False)
    iid = Column(Integer,primary_key=True,nullable=False)
    state = Column(String(7),nullable=False)
    adddate = Column(Date,nullable=False)
    rate = Column(Integer)
    tags = Column(String(500))

    def __repr__(self):
       return "<Record(name='%s', iid='%s', rate='%s')>" % (
                self.name, self.iid, self.rate)

def run():
    engine = create_engine("mysql+mysqldb://"+MYSQL_USER+":"+MYSQL_PASSWD+"@"+
                            MYSQL_HOST+"/"+MYSQL_DBNAME+
                            "?charset=utf8&use_unicode=0")
    Base.metadata.create_all(engine)

    # Session is a custom class
    Session = sessionmaker(bind=engine)
    session = Session()

    # Start query
    # Read from database and writes (uid, iid, rate) into a bin file
    ucnt=0
    usertable = dict()
    for q in session.query(Record.name).filter(and_(Record.typ=='anime',
        Record.rate!=None)).group_by(Record.name).all():
        usertable[q.name]=ucnt;
        ucnt+=1;

    icnt=0
    itemtable = dict()
    for q in session.query(Record.iid).filter(and_(Record.typ=='anime',
        Record.rate!=None)).group_by(Record.iid).all():
        itemtable[q.iid]=icnt
        icnt+=1

    partialquery = session.query(Record.name, Record.iid, Record.rate).filter(
                   and_(Record.rate!=None, Record.typ=='anime'));
    with open("../Data/input.bin","wb") as fw:
        strrec = struct.pack("ii", ucnt, icnt)
        fw.write(strrec)
        for username in usertable.iterkeys():
            for q in partialquery.filter(Record.name==username).all():
                strrec = struct.pack("iih", usertable[username],
                                     itemtable[q.iid], int(q.rate))
                fw.write(strrec)

###
# Test if valid
def test():
    import struct
    cnt=0;
    with open("../Data/input.bin","rb") as fr:
        fr.seek(8)
        while True:
            strrec = fr.read(10)
            if not strrec: break;
            else: cnt+=1

    from sqlalchemy import and_
    cnt2 = int(session.query(Record).filter(and_(Record.rate!=None,
               Record.typ=='anime')).count())

    assert(cnt==cnt2)

if __name__=='__main__':
    run()
