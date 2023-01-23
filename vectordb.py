from sqlite3 import connect
from annoy import AnnoyIndex

class VDB:
    def __init__(self, dbName: str, fEmbed, dimEmbed=768, nTrees=10, metric='dot'):
        """
        :param dbName: name of the database files
        :param fEmbed: vector embedding function : (str -> Tensor)
        :param dimEmbed: embedding dimension
        :param nTrees: number of trees in the random projection forest
        :param metric: "angular", "euclidean", "manhattan", "hamming", or "dot"
        """
        self.dimEmbed = dimEmbed
        self.nTrees = nTrees
        self.metric = metric
        self.dbName = dbName
        self.fEmbed = fEmbed
        # ANNOY
        self.annoy = AnnoyIndex(self.dimEmbed, self.metric)
        self.annoyIndexName = f'{self.dbName}.ann'
        # SQLITE
        self.sqliteDBName = f'{self.dbName}.db'
        self.sqliteConn = connect(self.sqliteDBName)
        self.cur = self.sqliteConn.cursor()
        self.cur.execute("DROP TABLE IF EXISTS vdb")
        self.cur.execute("CREATE TABLE vdb (id INTEGER PRIMARY KEY, item TEXT)")
    def build(self, items):
        """
        :param items: iterable of objects to be inserted
        :return: ()
        """
        for i, x in enumerate(items):
            self.cur.execute('INSERT INTO vdb VALUES (?, ?)', (i, x))
            xv = self.fEmbed(x)
            self.annoy.add_item(i, xv)
        self.sqliteConn.commit() # commit SQL transaction
        self.annoy.build(self.nTrees) # build Annoy index
        self.annoy.save(self.annoyIndexName) # save Annoy index
    def getKNN(self, x, k = 10):
        """
        :param x: query data
        :param k: number of NNs to retrieve
        :return:
        """
        self.annoy.load(self.annoyIndexName)
        v = self.fEmbed(x)
        ixResults = self.annoy.get_nns_by_vector(v, k)
        for ix in ixResults:
            c = self.cur.execute('SELECT item FROM vdb WHERE id = (?)', (ix,))
            yield c.fetchone()[0]
