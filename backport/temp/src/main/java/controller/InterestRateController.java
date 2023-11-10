package controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.IService;
import entity.InterestRate;
import mapper.InterestRateRepository;
import org.springframework.web.bind.annotation.*;
import service.rateServer;

import javax.annotation.Resource;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


@RestController
public class InterestRateController {

    @Resource
    private rateServer rateserver;
    @Resource
    private InterestRateRepository rateRepository;
    @RequestMapping("/helloo")
    public String hello(){return "Hello world!";}
    @RequestMapping("/interest")
    public String interestCal(/*@PathVariable String time*/){
//        QueryWrapper<InterestRate> intQueWrapper=new QueryWrapper<>();
//        double year= Double.parseDouble(time);
//        if(year==0.25)
//        {
//            intQueWrapper.eq("rate_type","三个月");
//        }
//        else if(year==0.5)
//        {
//            intQueWrapper.eq("rate_type","半年");
//        }
//        else if(year==1)
//        {
//            intQueWrapper.eq("rate_type","一年");
//        }
//        else if(year==2)
//        {
//            intQueWrapper.eq("rate_type","二年");
//        }
//        else if(year==3)
//        {
//            intQueWrapper.eq("rate_type","三年");
//        }
//        else if(year==5)
//        {
//            intQueWrapper.eq("rate_type","五年");
//        }
//        else
//        {
//            intQueWrapper.eq("rate_type","活期存款");
//        }
//        InterestRate intRate=rateRepository.selectOne(intQueWrapper);
//        String result=Double.toString(intRate.getRate());
//        return result;
            return "Hello world!";
    }

}
