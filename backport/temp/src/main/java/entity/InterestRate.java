package entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("interest_rate")
public class InterestRate {
    @TableId(type = IdType.AUTO)
    private double year;
    private double rate;

    public InterestRate() {
        // 默认构造函数
    }

    public InterestRate(double year, double rate) {
        this.year = year;
        this.rate = rate;
    }
    
    public double getYear() {
        return year;
    }

    public void setYear(double year) {
        this.year = year;
    }

    public double getRate() {
        return rate;
    }

    public void setRate(double rate) {
        this.rate = rate;
    }
}

