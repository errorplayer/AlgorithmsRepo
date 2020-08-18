import random


# reference
# http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/

class LDAmodel:
    def __init__(self, topic_num=10):
        self.topic_num = topic_num
        self.word2index = {}
        self.index2word = {}
        self.count = 0
        self.training_corpus_digit = []
        self.training_topics = []
        self.update_flag = 0
        self.dict_t_d = {}
        self.dict_w_t = {}
           
    def prob_t_d(self, topic_index, doc_index, topics, update_flag=1):
        if update_flag == 0:
          if self.dict_t_d.get(topic_index, 0) != 0:
            if self.dict_t_d[topic_index].get(doc_index, 0) != 0:
              return self.dict_t_d[topic_index][doc_index]

        denom = len(topics[doc_index])
        if not self.dict_t_d.get(topic_index, 0):
          d_ = {}
          d_[doc_index] = 0.0
          self.dict_t_d[topic_index] = d_
        if denom == 0:
            return 0.0
        count = 0
        for topic in topics[doc_index]:
            if topic == topic_index:
                count += 1
        self.dict_t_d[topic_index][doc_index] = count / denom
        return count / denom
    
    def prob_w_t(self, word_index, topic_index, topics, doc_digit, update_flag=1):
        if update_flag == 0:
          if self.dict_w_t.get(word_index, 0) != 0:
            if self.dict_w_t[word_index].get(topic_index, 0) != 0:
              return self.dict_w_t[word_index][topic_index]

        count_word = 0
        count_topic = 0
        for idx_doc, document in enumerate(doc_digit):
            for idx_word, word in enumerate(document):
                if word_index == word:
                    count_word += 1
                if topics[idx_doc][idx_word] == topic_index:
                    count_topic += 1
        if not self.dict_w_t.get(word_index, 0):
          d_ = {}
          d_[topic_index] = 0.0
          self.dict_w_t[word_index] = d_
        if count_topic == 0:
          return 0.0
        else:
          self.dict_w_t[word_index][topic_index] = count_word / count_topic
          return count_word / count_topic
        
    def train(self, traing_corpus, iters=5):
        for idx_doc, document in enumerate(traing_corpus):
            tmp_topic = []
            tmp_content_digit = []
            for idx_word, word in enumerate(document):
                # no record in our vocabulary, then add it.
                if not self.word2index.get(word, 0):
                    self.word2index[word] = self.count
                    self.index2word[self.count] = word
                    self.count += 1
                tmp_content_digit.append(self.word2index[word])
                tmp_topic.append(random.randint(0,self.topic_num-1))
            self.training_topics.append(tmp_topic)
            self.training_corpus_digit.append(tmp_content_digit)
        for i in range(iters):
          for idx_doc, document in enumerate(self.training_corpus_digit):
            for idx_word, word in enumerate(document):
              flag = 0
              if random.uniform(0, 1) > 0.95:
                flag = 1
              score_ = [self.prob_t_d(t, idx_doc, self.training_topics, flag)*self.prob_w_t(word, t, self.training_topics, self.training_corpus_digit, flag) for t in range(self.topic_num)]
              self.training_topics[idx_doc][idx_word] = score_.index(max(score_))
          
    def predict(self, test_corpus, iters=5):
        self.train(test_corpus, iters)
        res_ = []
        pointer_ = len(test_corpus)
        for idx_doc, document in enumerate(self.training_topics[-pointer_:]):
            total_len = len(document)
            tmp_topic_vec = [0.0  for i in range(self.topic_num)]
            for topic_i in document:
                tmp_topic_vec[topic_i] += 1
            res_.append(tmp_topic_vec)
        return res_
# texts = [doc2seglist(parag) for parag in tqdm(contents[:200])]  
text1 = '''
将五花肉切成单约2cm左右的块。
锅中放入适量的清水，并把五花肉倒进锅中，加入适量的料酒烧开烫五分钟。小编我之所以选择使用料酒，是为了更好的去除五花肉当中的肉腥味。大家...
接着往锅中加入适量的食用油，待油冒烟，倒入几块冰糖，搅拌融化。
把烫好的五花肉倒入锅中翻炒三分钟。
接着往锅中加入桂皮，八角，香叶，蒜头，生姜和大蒜段，翻炒三分钟。
'''   
text2 = '''
五花肉解冻到半硬，开始切片。
锅中倒水，将五花肉放入锅中，中火煮3分钟。
青椒洗净切斜刀段。
锅内不放油放入切好的肉片，翻炒至肉片卷曲，肥肉透明最好，这样猪肉的油都被炒出来了，吃起来不腻。
锅里加入菜籽油大火将锅烧热，加郫县豆瓣酱炒至出香味。
放入香芹段、洋葱块、蒜片。
再加入青椒块、五花肉翻炒一会儿。
最后加入料酒、白糖、盐、生抽翻炒均匀即可食用。
出锅装盘，配米饭很好吃。
肥而不腻，颜色养眼，是下饭菜的首选。   
'''
text3 = '''
传统体育竞技赛事中，从来不会缺少奇迹的身影，欧冠赛场上的“伊斯坦布尔神话”、“诺坎普奇迹”、NBA赛场上骑士1-3落后逆转夺冠……电子竞技作为体育竞技赛事的新贵，自然也不会缺少这样的戏码，在这次王者荣耀世界冠军杯的总决赛舞台上，TS与DYG就为大家带来了经典一战，TS在0-3落后的情况连赢四局，夺得冠军，这是首冠总决赛历史上首次出现“让三追四夺冠”的奇迹，这一场比赛注定载入史册。

　　体育竞技赛事最让人着迷的无疑就是不到最后一刻，永远无法预判结果，这种激烈的竞技性正是体育赛事的魅力所在，而电竞无疑完美传承了这一魅力，当TS以0-3落后的时候，大部分人都在猜测他们最终会以怎样的比分失利，但这群年轻的小伙子却用永不放弃的精神，最终创造奇迹，在体育史上画下了浓重的一笔。

　　世冠这“一赛”完美助力电竞北京。

　　本次世冠总决赛是北京市 “一会一展一赛” 三大活动中的“一赛”，TS与DYG用一场足以载入史册的经典对决，让电竞北京这“一赛”完美升华，让更多观众体会到了电竞赛事的独特魅力，也让他们对电竞的未来发展更加充满信心。

　　在体育赛事的发展历史中，一项赛事的成功，或者一场经典比赛的诞生，往往就可能为举办城市打上一个标签。

　　堪称经典的王者世冠总决赛，又适逢北京大力推广电子竞技的时机，天时地利俱在，电竞在北京这座首都城市的发展，真心令人期待！
'''

text4 = '''
北京时间8月16日，王者荣耀世界冠军杯总决赛，TS在先输三局的情况下连扳四局，完成史诗级逆转，以4-3战胜DYG，拿下本届世冠总冠军，这也是王者荣耀职业联赛历史以来，首次在决赛中出现让三追四夺冠的壮举。


　　首局比赛，DYG出人意料的选出瑶辅助，收获奇效，打出5-0-15的完美数据。比赛开局后，DYG中野辅直接强势入侵野区，拿下3Buff开局，随后双方竞争激烈，TS三人包上路拿下关羽一血并推掉一塔，但随后久诚沈梦溪赶到，收掉暖阳兰陵王帮助DYG止损；比赛中期，DYG逐渐掌控比赛节奏，进入强势期的公孙离在瑶的保护下输出极高，TS在输掉几波小团战后，只能频频避战，让掉一些资源，但他们也借此拖缓了DYG的进攻脚步；比赛进入大后期，TS找到机会，击杀深入的湘军，随后拿下风暴龙王并推掉DYG中路高地；最后一波团战，清清的关羽成为关键先生，直接切到诗酒孙尚香，帮着DYG打出1换5团战，一波终结首局比赛。


　　第二局比赛，DYG帮助久诚选下元歌。DYG从开局开始便进入状态，久诚的元歌前期连续击杀暖阳赵云，帮助DYG拿到赛场话语权；14分10秒，清清的芈月与久诚的元歌相继完成越塔击杀，DYG乘胜推进，少人的TS只能尽力防守，但易峥蒙犽压泉水疯狂输出，最终DYG用时15分钟，拿下第二局。


　　第三局比赛，失利就要被DYG拿到赛点的TS放手一搏，暖阳裴擒虎前中期节奏出色，TS很快拿到经济优势，而DYG则在陷入劣势后选择避战；比赛进入后期，DYG拖时间的目的达到，久诚百里守约装备成型，DYG形成反压制，久诚百发百中的百里守约完全压制了TS，DYG不断向前推进，反超经济；最后一波团战，DYG压迫TS高地，久诚远程火力支撑，DYG打出1换4，推掉TS水晶，3-0拿到赛点。


视频-久诚百里守约百发百中。
　　第四局比赛，TS背水一战，千世选出他世冠至今百分百胜率的不知火舞，DYG则帮助清清拿到马超。比赛前中期，DYG凭借马超的单线优势以及游走取得优势；比赛第12分钟，阿豆关键开团背回易峥，TS找回场上主动权；最后一波团战，神人老夫子灵性蹲草丛绑住刘邦，TS集火完成秒杀，随后拿下暴君，借助兵线强行一波结束比赛，扳回一局。


　　第五局比赛，TS帮助诗酒拿下百里守约，4无尽百里创始人在绝境中选出了自己的招牌。进入比赛后，尽管诗酒前期命中率一般，但TS依旧建立起经济优势；比赛中期，诗酒逐渐找到手感，连续狙中引得赛场阵阵欢呼，虽然DYG一度凭借久诚一波天秀秒杀百里的操作扳回一些局面，但在后期，他们仍然对诗酒的百里守约毫无办法，最终在百里的火力压制之下，TS再次扳回一局。


　　第六局比赛，连扳两局的TS势头正盛，仅用时12分41秒便再取一胜，暖阳选出韩信拿下超神战绩，TS连扳三局，大比分战至3-3，双方进入巅峰对决。


　　巅峰对决，DYG果断为久诚选下百分百胜率的百里守约，同时选到露娜和猪八戒的双打野体系；TS则为暖阳选到镜，诗酒选出招牌马可波罗，阿豆果断选到盾山。比赛前期，TS利用百里还没有成型的时间连续找到节奏，很快在人头上取得3-0领先，并推掉上路一塔；比赛中期，TS依旧掌握场上局面，人头比来到6-0；比赛进入10分钟以后，久诚的百里守约开始发力，连续狙中帮助易峥收下神人猪八戒的人头；后期比赛，久诚的百里守约虽然仍弹无虚发，但几波强开团，诗酒的马可波罗都发挥完美，连续帮助TS打赢团战；最后阶段，TS集结推掉DYG中路高地，DYG则选择让清清单带掉了TS的下路高地；最后一波，TS整理后推进，阿豆关键开团开到久诚，虽然久诚闪现拉开，但距离被拉近后的百里守约威胁大减，TS击杀DYG数人，一波终结比赛，完成让三追四，拿下冠军。
'''

def generator_batch(inputs, batch_size=50):
  tl = len(inputs)
  for i in range(tl//batch_size):  
    yield inputs[i*batch_size:i*batch_size+batch_size] 
  if (tl//batch_size)*batch_size != tl:
    yield inputs[(tl//batch_size)*batch_size:]  

texts = [doc2seglist(text1), doc2seglist(text2), doc2seglist(text3), doc2seglist(text4)]
lda = LDAmodel(10)
for idx, texts_batch in enumerate(tqdm(generator_batch(texts, 1))):
 lda.train(texts_batch, iters=10)

result = lda.predict(texts[2:], iters=10)
for i in result:
  print(i)
