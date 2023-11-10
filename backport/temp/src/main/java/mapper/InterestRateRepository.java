package mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import entity.InterestRate;
import org.apache.ibatis.annotations.Mapper;
import com.baomidou.mybatisplus.extension.service.IService;
import org.apache.ibatis.annotations.Select;
import org.springframework.web.bind.annotation.RequestMapping;

import java.util.List;

@Mapper
public interface InterestRateRepository extends BaseMapper<InterestRate> {
//    @Select("select rate from interest_rate where interest_type = #{type}")
//    List<Double> findAll();
}

