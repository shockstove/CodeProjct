package com.example.demo.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.demo.entity.InterestRate;
import com.example.demo.entity.Result;
import com.example.demo.mapper.InterestRateRepository;
import com.example.demo.mapper.ResultRepository;
import com.example.demo.service.ResultService;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import com.example.demo.service.rateServer;

import javax.annotation.Resource;
import java.util.List;


@Controller
public class calController {

    @Resource
    private ResultRepository ResultRepository;
    @Resource
    private ResultService ResultService;
    @GetMapping("/save")
    @CrossOrigin(originPatterns = "*", methods = {RequestMethod.GET, RequestMethod.POST})
    @ResponseBody
    public void resSave(@RequestParam("formula") String formula,@RequestParam("result")String result){
        ResultService.save(new Result(formula,result));
    }
    @GetMapping("/history")
    @CrossOrigin(originPatterns = "*", methods = {RequestMethod.GET, RequestMethod.POST})
    @ResponseBody
    public List<Result> GetHistory(){
        QueryWrapper<Result> resultQueryWrapper=new QueryWrapper<>();
        return ResultService.list();
    }
}
