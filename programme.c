

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


// 深度优先搜索，广度优先搜索
// 二叉排序树和平衡二叉树
// b-树b+树
// 哈希表