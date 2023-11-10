#include <iostream>
using namespace std;
class Rectangle {
	friend class Rectangular;
public:
	Rectangle() : length(0), width(0) ,s(0){};
	Rectangle(const Rectangle& p)
	{
		if (this == &p)return;
		length = p.length;
		width = p.width;
		s = p.s;
	}
	Rectangle& operator=(const Rectangle& p)
	{
		if (this == &p) return *this;
		length = p.length;
		width= p.width;
		s = p.s;
		return *this;
	}
	void setlenth(int l) {
		length = l;
		cout << "已设置当前的长为" << length << endl;
	}
	void setwidth(int w) {
		width = w;
		cout << "已设置当前的宽为" << width << endl;
	}
	void square() {
		 s = length * width;
		cout << "长方形的面积为" << s << endl;

	};
	void test01() { cout << s << endl; };
private:
	int length;
	int width;
 int s;
};
class Rectangular:public Rectangle {
public:
	Rectangular() :height(0), v(0) {};
	Rectangular(const Rectangular& p)
	{
		if (this == &p)return;
		height = p.height;
		v = p.v;
	}
	Rectangular& operator=(const Rectangular& p)
	{
		if (this == &p) return *this;
		height = p.height;
		v = p.v;
		return *this;
	}
	void setheight(int h) {
		height = h;
		cout << "已设置当前的高为" << height << endl;
	};
	void volume() {
            square();
			v = height * s;
		cout << "长方体的体积为" << v<< endl;
		
	};
	private:
		int height;
		int v;

};
int main() {
	Rectangle r1;
	Rectangular r2;
	
	r1.setlenth(4);
	r1.setwidth(5);
	r1.square();
	r1.test01();
	r2.test01();
	r2.setheight(9);
    r2.setlenth(4);
    r2.setwidth(5);
	r2.volume();
	
	
	system("pause");
	
	return 0;
}