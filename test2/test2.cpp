#include<iostream>
#include<string>
using namespace std;

int main( )
{
    char a[10006];
    cin>>a;
    int k;
    cin>>k;
    int maxNum = 0;
    bool b[10006]={0};
    int i=0;
    for(;a[i]!='\0';i++)
    {
        if(a[i]=='a'||a[i]=='e'||a[i]=='i'||a[i]=='o'||a[i]=='u'){
            b[i]=1;
        }
        else b[i]=0;
    }
    for(int m=0;m<i-k-1;m++)
    {
        int temp=0;
        for(int j=m;j<k+m;j++){
            temp+=b[j];
        }
        if(temp>maxNum){
            maxNum=temp;
        }
    }
    cout<<maxNum<<endl;
    return 0;
}