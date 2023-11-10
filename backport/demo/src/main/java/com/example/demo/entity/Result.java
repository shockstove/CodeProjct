package com.example.demo.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("cal")
public class Result {
    public String getFormula() {
        return formula;
    }

    public void setFormula(String formula) {
        this.formula = formula;
    }

    public String getResult() {
        return result;
    }

    public void setResult(String result) {
        this.result = result;
    }

    public Result(String formula, String result) {
        this.formula = formula;
        this.result = result;
    }

    private String formula;
    private String result;
}
