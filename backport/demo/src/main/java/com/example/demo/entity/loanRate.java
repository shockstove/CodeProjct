package com.example.demo.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("loan_rate")
public class loanRate {
    @TableId(type = IdType.AUTO)
    private String rate_type;
    private double rate;

    public void setRate_type(String rate_type) {
        this.rate_type = rate_type;
    }

    public void setRate(double rate) {
        this.rate = rate;
    }

    public String getRate_type() {
        return rate_type;
    }

    public double getRate() {
        return rate;
    }

    public loanRate(String rate_type, double rate) {
        this.rate_type = rate_type;
        this.rate = rate;
    }
}
