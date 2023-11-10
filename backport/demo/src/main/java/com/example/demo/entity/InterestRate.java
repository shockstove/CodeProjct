package com.example.demo.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("interest_rate")
public class InterestRate {
    @TableId(type = IdType.AUTO)
    private String rate_type;
    private double rate;

    public InterestRate() {
        // 默认构造函数
    }

    public InterestRate(String rate_type, double rate) {
        this.rate_type = rate_type;
        this.rate = rate;
    }
    
    public String getRate_type() {
        return rate_type;
    }

    public void setRate_type(String rate_type) {
        this.rate_type = rate_type;
    }

    public double getRate() {
        return rate;
    }

    public void setRate(double rate) {
        this.rate = rate;
    }
}

