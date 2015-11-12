

// 二叉树性质
// 1、第i 上至多有2ⁿ-¹个结点
// 2、深度为n的二叉树至多有2ⁿ-1个结点
// 3、对任何一棵二叉树T，如果其终端接点数为n0，度为2的结点数为n2，则n0 = n2 + 1  （n=n0+n1+n2,n=n1+2n2+1）(设B为分支数，n=B+1,B=n1+2n2)
// 4、具有n个结点的完全二叉树的深度为(取整㏒2（n）)+1
// 5、在含有n个结点的二叉链表中有n+1个空链域
//二叉树  非递归中序遍历
//1.栈顶指针不为空，遍历左子树
//2.栈顶指针为空，若是左子树返回，访问当前层即栈顶指针所指根节点
//3.若是右子树返回，表明当前层遍历结束，退栈
Status InOrderTraverse(BitTree T,Status (*Visit)(TElemType e)){
  InitStack(S); p = T;
  while(p || !StackEmpty(S)){
      if(p){Push(S,p); p = p->lchild; }//根指针进栈，遍历左子树
      else{          //根指针退栈，访问根节点，遍历右子树
      	Pop(S,p);if( !Visit(p->data)) return ERROR;
      	p = p->rchild;
      }
  }
  return OK;

}

//遍历二叉树中，建立存储结构
//先序建立二叉树的二叉链表
//ABC**DE*G**F***   *代表空， 输入完全二叉树的形式，否则无法识别
Status CreateBiTree(BitTree &T){
	//按先序输入二叉树结点的值，*表示空树
	scanf(&ch);
	if(ch == '') T = NULL;
	else{
		if( !( T= (BiTNode *)malloc(sizeof(BiTNode)))) exit(OVERFLOW);
		T -> data = ch;                   //生成根结点
		CreateBiTree(T->lchild);          //构造左子树     
		CreateBiTree(T->rchild);          //构造右子树
	}
	return OK;
}

//遍历二叉树基本操作是访问结点，即不论按哪一种次序进行遍历
//对含n个结点的二叉树，时间复杂度均为O(n)
//所需辅助空间为遍历过程中栈的最大容量，即树的深度，最坏情况下为n，空间复杂度也为O(n)


// 线索二叉树  增加一头结点，好比建立了一个双向线索链表
// 双向线索链表为存储结构时对二叉树进行遍历的算法
typedef enum PointerTag{Link, Thread};//Link==0;指针,Thread == 1,线索
typedef struct BitThrNode
{
	TElemType data;
	struct BiThrNode *lchild, *rchild;
	PointerTag LTag,RTag;
}BitThrNode, *BiThrTree;

Status InOrderTraverse_Thr(BitTree T, Status(*Visit)(TElemType e)){
	//T指向头结点，头结点的左链lchild指向根节点
	p=T->lchild;
	while( p!=T ){   //空树或遍历结束时，p=T
		while(p->LTag == Link)p=p->lchild;
		if(!Visit(p->data))return ERROR;
		while(p->RTag==Thread&&p->rchild!=T){
			p=p->rchild;Visit(p->data); 
		}
		p=p->rchild;

	}
	return OK;
}

//二叉线索化
// 附设一个指针pre始终指向刚刚访问过得结点，若指针p指向当前访问的结点，则pre指向它的前驱
Status InOrderTreading(BitThrTree & Thrt,BitThrTree T){
	//中序遍历二叉树T，并将其中序线索化，Thrt指向头结点
	if (!(Thrt = (BiThrTree)malloc(sizeof(BiThrNode))))exit (OVERFLOW);
	Thrt->LTag = Link;  Thrt->RTag = Thread;   //建立头结点
	Thrt->rchild = Thrt;    //右指针回指
	if(!T)Thrt->lchild = Thrt;   //若二叉树为空，则空指针回指
    else{
    	Thrt->lchild = T; pre = Thrt;
    	InThreading(T);//中序遍历进行中序线索化
    	pre->rchild = Thrt; pre->RTag = Thread;//最后一个结点线索化
    	Thrt->rchild = pre;
    }
    return OK;
}

void InThreading(BiThrTree p){
	if(p){
		InThreading (p->lchild);  //左子树线索化
		if(!p->lchild){p->LTag = Thread; p->lchild = pre;}//前驱线索
		if(!pre->rchild)(pre->RTag = Thread; pre->rchild = p;)//后继线索
		pre = p;  //保持pre指向p的前驱
		InThreading(p->rchild);//右子树线索化
	}
}
// 递归的条件
// 1、可以把要解决的问题转化为一个新问题，而这个新的问题的解决方法仍与原来的解决方法相同，只是所处理的对象有规律地递增或递减。
// 2、可以应用这个转化过程使问题得到解决。
// 3、必定要有一个明确的结束递归的条件。


//哈夫曼树
//最优树，带权路径长度最短的树
// 从树中一个结点到另一个结点之间的分支构成这两个结点之间的路径，路径上的分支数目称路径长度。
// 树的路径长度是从树根到每一结点的路径长度之和，完全二叉树是这种路径长度最短的二叉树。  
// 推广，考虑带权的特点，结点的带权路径长度为该节点到树根之间的路径长度与结点上权的乘积。
// 树的带权·路径长度为树中所有“叶子”结点的带权路径长度之和。
// 带权路径长度最小的二叉树叫做最优二叉树或哈夫曼树。
//实际作用，比如将学生成绩化为5个等级机制，传统<60,<80,<90,如果60分以下人很少，总体人数多，会造成比较次数过多,所以要考虑带权情况

// 哈弗曼算法->构建哈夫曼树
// 1、根据给定的n个权值构成n棵二叉树集合F，其中每个二叉树Ti中只有一个带权为Wi的根节点，左右子树均空。
// 2、在F中选取两颗根节点的权值最小的树作为左右子树构造一棵新的二叉树，且置新的二叉树根节点权值为左右子树上根节点权值之和
// 3、在F中删除这两棵树，同时将新得到的二叉树加入F中.
// 4、重复2,3，直到F只含一棵树为止，这棵树便是哈夫曼树。

//频率高的字符编码长度变短，总长便可变小。
//若要设计长短不等的编码，则必须任意一个字符的编码都不是另一个字符编码的前缀，这种编码称前缀编码
// 左分支表示0，有分支表示1，根节点到叶子结点路径上分支字符组成的字符串作为叶子节点字符的编码，得到的比为二进制前缀编码。

// 哈弗曼编码具体做法：
// 一棵有n个叶子节点的哈夫曼树共有2n-1个结点，可以存储在一个大小为2n-1的一位数组中。由于在构成哈夫曼树之后，为求编码需从
// 叶子结点出发走一条叶子到根的路径，而译码需从根出发，走一条根到叶子的路径。
// 则对每个结点而言，既须知双亲的信息，有须知孩子结点的信息。
/*-------哈夫曼树和哈弗曼编码的存储表示-------------------------------------*/
typedef struct 
{
	unsigned int weight;
	unsigned int parent,lchild,rchild;
}HTNode,  *HuffmanTree;  //动态分配数组存储哈夫曼树

typedef char * *HuffmanCode;//动态分配数组存储哈弗曼编码表

void HuffmanCoding(HuffmanTree &HT, HuffmanCode &HC, int *w, int n){
	//w存放n个字符的权值，构造哈夫曼树HT，并求出n个字符的哈弗曼编码HC
	if(n<=1)return;
	m=2*n-1;
	HT - (HuffmanTree)malloc((m+1)*sizeof(HTNode));//0号单元未用
	for(p=HT,i=1;i<=n;++i,++p,++w) *p = {*w,0,0,0};
	for(;i<=m;i++,p++) *p = {0,0,0,0};
	for (i = n+1; i <= m; ++i)//建哈夫曼树
	{
		//在HT[1...i-1]选择parent为0且weight最小的两个结点，其序号分别为s1,s2
		Select(HT,i-1,s1,s2);
		HT[s1].parent = i; HT[s2].parent = i;
		HT[i].lchild = s1; HT[i].rchild = s2;
		HT[i].weight = HT[s1].weight + HT[s2].weight;
	}
    //-------从叶子到根逆向求每个字符的哈弗曼码----------------------------
    HC = (HuffmanCode)malloc((n+1)*sizeof(char *));//分配n个字符编码的头指针向量
    cd = (char *)malloc(n*sizeof(char));//分配求编码的工作空间
    cd[n-1] = "\0";//编码结束符
    for(i = 1;i<=n;i++){  //逐个字符求哈弗曼编码
         start = n-1;  //编码结束符位置
         for (c=i,f=HT[i].parent;f!=0; c=f,f=HT[f].parent)//从叶子到根逆向求编码
         {
         	if(HT[f].lchild == c) cd[--start] = "0";
         	else cd[--start] = "1";
         }
         HC[i] = (char *)malloc((n-start)*sizeof(char));//为第i个字符编码分配空间
         strcpy(HC[i],&cd[start]);//从cd复制编码到HC
    }
    free(cd);//释放工作空间
}

//从根出发-----无栈非递归遍历哈夫曼树，求哈弗曼编码
HC = (HuffmanCode)malloc((n+1)*sizeof(char *));
p=m;  cdlen = 0;
for(i=1;i<=m;++i) HT[i].weight = 0;//遍历哈夫曼树时用作结点状态标志
while(p){
	if(HT[p].weight == 0){  //向左
       HP[p].weight = 1;
       if(HT[p].lchild!=0){p=HT[p].lchild; cd[cdlen ++] = "0"; }
       else if (HT[p].rchild   == 0)       //登记叶子结点的字符的编码
       {
       	  HC[p] = (char *)malloc((cdlen + 1)*sizeof(char);
       	  cd[cdlen] = "\0";  strcpy(HC[p],cd);  //复制编码
       }
	}
	else if (HT[p].weight == 1)//向右
	{
		Ht[p].weight = 2;
		if(HT[p].rchild != 0 ){p=HT[p].rchild; cd[cdlen++] = "1";}
	}else{     //HT[p].weight == 2，退回
        HT[p].weight = 0; p=HT[p].parent; --cdlen;//退到父结点，编码长度减1
	}
}//译码过程是分解电文中字符串，从根出发，按字符‘0’或‘1’确定找左孩子或右孩子，直至叶子结点，便求得该子串相应字符。


//不是根据某种确定的计算法则，而是试探和回溯的搜索技术求解——回溯法，如八皇后问题
/*注：八皇后问题，是一个古老而著名的问题，是回溯算法的典型案例。该问题是国际西洋棋棋手马克斯·贝瑟尔于1848年提出：
在8×8格的国际象棋上摆放八个皇后，使其不能互相攻击，即任意两个皇后都不能处于同一行、同一列或同一斜线上，
问有多少种摆法*/
//求四皇后问题的所有合法布局（八皇后问题简化版）
//这是一棵四叉树，树上每个结点表示局部布局或一个完整的布局，根节点无任何棋子，每个棋子都有4个可选择的位置，
//但在任何时刻，棋盘的合法布局都必须满足三个约定:任何两个棋子都不占据棋盘上的同一行，或同一列或一对角线.
//求所有合法布局的过程即为在上述约定条件下先根遍历的过程
//遍历中访问结点的操作是：判别棋盘上是否已得到一个完整布局，若是，在输出该布局，否则一次先根遍历满足约束条件的
// 各棵子树，即首先判断该子树的布局是否合法，若合法，遍历该子树，否则剪去该子树分支
void Trial(int i, int n){
	//进图本函数时，在n*n棋盘前i-1行已放置互不攻击的i-1个棋子
	//i>n,求得一个合法布局
	if(i>n)  Output(B);
	else for(j=1;j<=n;++j){
         //在第i行j列放置一个棋子
		if(当前局合法)Trial(i+1,n);
		移走第i行j列棋子；
	}
}



//图
//图中数据元素叫顶点，<v,w>∈VR(顶点关系集合)  ,表示从v到w的一条弧，v叫弧尾或初始点，w叫弧头或终端点，此时图为有向图
// <v,w>∈VR必有<w,v>∈VR，对称，表示v到w的一条边，此时图为无向图
// 有1/2n(n-1)条边的无向图称为完全图，有n（n-1）条弧的有向图称有向完全图，带权的图称网
//无向图中，度为顶点和边相关联的数目，有向图，分出度和入度
// 路径是一个顶点序列，路径的长度是路径上边或弧的数目，序列中顶点不重复出现的路径称简单路径。
// 除了第一个顶点和最后一个顶点之外，其余顶点不重复出现的回路，称简单回路或简单环。
// 无向图中，顶点a到b有路径，称a到b是连通的。对于图中任意两个顶点都是连通的，称图为连通图
// 无向图中的极大连通子图叫连通分量。
// 有向图中，对于a，b，a到b，b到a都存在路径，图为强连通图，有向图中的极大强连通子图称强连通分量。
// 一个连通图的生成树是一个极小连通子图
// **一棵有n个顶点的生成树有且仅有n-1条边，如果一个图有n个顶点和小于n-1条边，则是非连通图，如果多余n-1条边，则一定有环。但有n-1条边的图不一定是生成树。

//图的存储结构
//数组表示法-矩阵
#define INFINITY INT_MAX //最大值，无穷大
#define MAX_VERTEX_NUM 20 //最大顶点个数
typedef enum {DG,DN,UDG,UDN}GraphKind;//{有向图，无向图，无向图，无向网}
typedef struct ArcCell
{
	VRType  adj;//VRType是顶点关系类型，对无权图，用1或0表示相邻否，对带权图，则为权值类型
	InfoType *info;//该弧相关信息的指针
}ArcCell,AdjMatrix[MAX_VERTEX_NUM][MAX_VERTEX_NUM];

typedef struct 
{
	VertexType  vexs[MAX_VERTEX_NUM];  //顶点向量
	AdjMatrix arcs;  //邻接矩阵
	int vexnum,arcnum;  //图的当前顶点数和弧数
	GraphKind kind;//图的种类标志
}MGraph;

//采用数组（邻接矩阵）表示法，构造无向网
Status CreateUDN(MGraph &G){
	scanf(&G.vexnum,&G.arcnum,&IncInfo);//IncInfo为0则各弧不含其它信息
	for(i=0;i<G.vexnum;++i)scanf(&G.vexs[i]);//构造顶点向量
	for(i=0;i<G.vexnum;++i)//初始化邻接矩阵
		for(j=0;j<G.vexnum;++j)G.arcs[i][j] = {INFINITY,NULL};
	for(k=0;k<G.arcnum;++k){  //构造邻接矩阵
		scanf(&v1,&v2,&w);   //输入一边依附的顶点和权值
		i = LocateVex(G,v1);  j = LocateVex(G.v2);//确定v1，v2在G中的位置
		G.arcs[i][j].adj = w;//弧<v1,v2>的权值
		if(IncInfo)Input(*G.arcs[i][j].info);//若弧有相关信息，则输入
		G.arcs[j][i] = G[i][j];
	}  
	return OK;
}

//邻接表示图的一种链式存储结构
//邻接表中，对图中每个顶点建立一个单链表，第i个单链表中的结点表示依附于顶点vi的边
// 结点由3个域组成，邻接域adjvex表示顶点vi邻接的点在图中的位置。
// 链域nextarc表示下一条边或弧的结点。
// 数据域info存储和边或弧相关的信息，如权值。
// 表头结点通常以顺序结构的形式荀淑，以便随意访问任意顶点的链表。

//图的邻接表存储表示
#define MAX_VERTEX_NUM 20
typedef struct ArcNode
{
	int adjvex;  //该弧所指向顶点位置
	struct ArcNode *nextarc;  //指向下一条弧的指针
	InfoType *info; //该弧相关信息的指针
}ArcNode;
typedef struct VNode
{
	VertexType data;//顶点信息；
	ArcNode *firstarc;//指向第一条依附该顶点的弧的指针
}VNode,AdjList[MAX_VERTEX_NUM];
typedef struct
{
	AdjList vertices;
	int  vexnum,arcnum; //图的当前顶点树和弧度数
	int kind;//图的种类
};
//若无向图中有n个顶点，e条边，则它的邻接表需n个头结点和2e个表结点，在边稀疏（e<<n(n-1)/2）的情况下，邻接表比邻接矩阵节省存储空间


//十字链表是有向图的一种链式存储结构，可以看成是有向图的邻接表和逆邻接表结合起来的一种表
//弧结点 tailvex,headvex,(尾域，头域)hlink（弧头相同的下一条弧）,tlink（胡尾相同下一条弧）,info
//顶点结点 data,firstin,firstout
//--------有向图的十字链表存储表示---------------------
#define MAX_VERTEX_NUM 20
typedef struct ArcBox
{
	int tailvex,headvex;  //该弧的尾和头结点的位置
	struct ArcBox *hlink,*tlink; //分别为弧头相同和弧尾相同的弧的链域
	Infotype *info;//该弧相关信息的指针
}ArcBox;

typedef struct VexNode
{
	VertexType data;
	ArcBox *firstin,*firstout;//分别指向该顶点的第一条入弧和出弧
}VexNode;
typedef struct 
{
	VexNode xlist[MAX_VERTEX_NUM];//表头向量
	int vexnum,arcnum;  //有向图的当前顶点数和弧数
}OLGraph;

Status CreateDG(OLGraph &G){
	//采用十字链表存储表示，构造有向图G
	scanf(&G.vexnum,&G.arcnum,&IncInfo);//IncInfo为0表示不含其他信息
    for (int i = 0; i < G.vexnum; ++i)//构造表头向量
    {
    	scanf(G.xlist[i].data);//输入顶点值
    	G.xlist[i].firstin = NULL;G.xlist[i].firstout = NULL;//初始化指针
    }
    for (k=0; k < G.arcnum; ++k)//输入各弧并构造十字链表
    {
    	scanf(&va,&v2);//收一条弧的始点和终点
    	i=LocateVex(G,v1);j=LocateVex(G,v2);//确定v1和v2在G中位置
    	p=(ArcBox *)malloc(sizeof(ArcBox));//假定有足够空间
    	*p = {i,j,G.xlist[j].firstin,G.xlist[i].firstout,NULL}//对弧结点赋值
    	G.xlist[j].firstin = G.xlist[i].firstout = p;//完成在入弧和出弧链头的插入
    	if(IncInfo) Input(*p->info);//带有弧相关信息，则输入
    }
}


//邻接多重表是无向图的一种链式存储结构
// 邻接多重表中，每一条边用一个结点表示
//边：mark(标志域,标记该条边是否被搜索过),ivex(顶点图中位置),ilink(下一条依附于顶点的边),jvex,jlink,info
// 顶点：data(顶点相关信息),firstedge(指示第一条依附于顶点的边)
//对无向图而言，邻接多重表和邻接表的差别，仅仅在于同一条边在临界表中用两个结点表示（重复的边，不同表示），而在邻接多重表中只有一个。
//------无向图的邻接多重表存储表示------------------------------
#define  MAX_VERTEX_NUM 20
typedef enum {unvisited,visited}VisitIf;
typedef struct EBox
{
	VisitIf mark;//访问标记
	int ivex,jvex;//该边依附的两个顶点的位置
	struct EBox *ilink,*jlink;//分别指向依附着两个顶点的下一条边
	InfoType *info;//该边信息指针
}EBox;
typedef struct VexBox
{
	VertexType data;
	EBox *firstedge;//指向第一条依附该顶点的边
}VerBox;
typedef struct 
{
	VerBox adjmulist[MAX_VERTEX_NUM];
	int vexnum,edgenum;//无向图的当前顶点数和边数
}AMLGraph;

//图的遍历——求解图的连通性，拓扑排序和求关键路径等算法的基础
// 深度优先搜索，广度优先搜索——对无向图和有向图都适用

//深度优先遍历——类似树的先根遍历
//假设初始状态是图中所有顶点都未曾被访问，则深度优先搜索可从图中某个顶点v出发，访问此顶点，然后依次从
// v的未被访问的邻接点出发深度优先遍历图，直至图中所有和v有路径相通的顶点都被访问到，若此时图中尚有顶点未被
// 访问到，则另选图中一个未曾被访问的顶点做起始点，重复上述过程，直至图中所有顶点都被访问到为止。
Boolean visited[MAX];//访问标志数组
Status(* VisitFunc)(int v);//函数变量

void DFSTraverse(Graph G, Status(*Visit)(int v)){
	//对图G作深度优先遍历
	VisitFunc = Visit;  //适用全局变量VisitFunc，是DFS不必设函数指针参数
	for(v=0;v<G.vexnum;++v)visited[v] = FALSE;//初始化
	for(v=0;v<G.vexnum;++v)
		if(!visited[v])DFS(G,v)//对尚未访问顶点调用DFS
}

void DFS(Graph G, int v){
	//从第v个顶点出发递归的深度优先遍历图G
	visited[v] = TRUE;  VisitFunc(v);//访问第v个结点
	for(w=FirstAdjVex(G,v);w>=0;w=NextAdjVex(G,v,w))
	   if(!visited[w])DFS(G,w); //对v尚未访问的邻接顶点w递归调用DFS    
}
//当用二位数组表示邻接矩阵作图的存储结构时，查找每个顶点的邻接点所需时间为O(n²)，n为顶点数
// 当以邻接表作图的存储结构时，找邻接点所需时间为O(e),e为无向图中边的树或有向图中弧的数。
// 当以邻接表作存储结构时，深度优先搜索遍历图的时间复杂度为O(n+e)




//广度优先搜索  ——类似于树的按层次遍历过程
// 从图中某顶点v出发，在访问了v之后依次访问v的哥哥未曾访问的邻接点，然后分别从这些邻接点出发依次访问它们的邻接点，
// 并使“先被访问的顶点的邻接点”先于“后被访问的顶点的邻接点”被访问，直至图中所有一杯访问的顶点的邻接点都被访问到。
// 若此时图中尚有顶点未被访问，则另选图中一个未曾被访问的顶点作起始点，重复上述过程，直至图中所有顶点都被访问到为止。
// 换句话或，广度优先搜索遍历图的过程是以v为起始点，由近至远，依次访问和v有路径相通且路径长度为1,2，。。的顶点。
//——--广度优先算法-------------------
void BFSTraverse(Graph G,Status (*Visit)(int v)){
  //按广度优先非递归遍历图G。适用辅助队列Q和访问标志数组visited
   for(v=0;v<G.vexnum;++v) visited[v] = FALSE;
   InitQueue(Q);  //置空的辅助队列Q
   for(v=0;v<G.vexnum;++v){
   	   if(!visited[v]){//v尚未访问
          visited[v] = TRUE;Visit(v);
          EnQueue(Q,v); //v入队列
          while(!QueueEmpty){
          	DeQueue(Q,u); //队头元素出队并置为u
          	for(w=FirstAdjVex(G,u);w>=0;w=NextAdjVex(G,u,w)){
          		if(!visited[w]){
          			visited[w] = TRUE;Visit(w);
          			EnQueue(Q,w);
          		}
          	}
          }
   	   }
   }
} 
//每个顶点至多进一次队列，遍历图的过程实质上是通过边或弧找邻接点的过程
//广度优先搜索遍历和深度优先搜索遍历时间复杂度相同，不同的知识对顶点的访问顺序不同

// 二叉排序树和平衡二叉树
// b-树b+树
// 哈希表