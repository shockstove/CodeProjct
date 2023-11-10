package com.example.demo.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.demo.entity.InterestRate;
import com.example.demo.mapper.InterestRateRepository;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import com.example.demo.service.rateServer;

import javax.annotation.Resource;


@Controller
public class InterestRateController {

    @Resource
    private rateServer rateserver;
    @Resource
    private InterestRateRepository rateRepository;
//    @RequestMapping("/helloo")
//    @ResponseBody
//    public String hello(){return "Hello world!";}
    @GetMapping("/interest")
    @CrossOrigin(originPatterns = "*", methods = {RequestMethod.GET, RequestMethod.POST})
    @ResponseBody
    public String interestCal(@RequestParam("time") String time){
        QueryWrapper<InterestRate> intQueWrapper=new QueryWrapper<>();
        double year= Double.parseDouble(time);
        if(year==0.25)
        {
            intQueWrapper.eq("rate_type","三个月");
        }
        else if(year==0.5)
        {
            intQueWrapper.eq("rate_type","半年");
        }
        else if(year==1)
        {
            intQueWrapper.eq("rate_type","一年");
        }
        else if(year==2)
        {
            intQueWrapper.eq("rate_type","二年");
        }
        else if(year==3)
        {
            intQueWrapper.eq("rate_type","三年");
        }
        else if(year==5)
        {
            intQueWrapper.eq("rate_type","五年");
        }
        else
        {
            intQueWrapper.eq("rate_type","活期存款");
        }
        InterestRate intRate=rateRepository.selectOne(intQueWrapper);
        String result=Double.toString(intRate.getRate());
        return result;
    }
}
