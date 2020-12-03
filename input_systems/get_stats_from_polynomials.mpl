GetStats := proc(polysys, vars, max_exp:=2)
local idx, jdx, j, LL, LLL, spol, reduct, num_zero_reduct, num_monoms:


MonomialList := e->`if`(type(e,`+`), convert(e,list), [e]):

for j from 1 to max_exp^numelems(vars) do
  num_monom := 0:
  num_zero_reduct:=0:
  LL := convert(j-1, base, max_exp):
  LLL := [op(LL), seq(0, i=1..numelems(vars) - nops(LL))] +~ 1:
  polys_subs := map2(subs, {seq(vars[i]=vars[i]^LLL[i], i=1..numelems(vars))}, polysys);
  for idx from 1 to nops(polys_subs)-1 do
    for jdx from idx+1 to nops(polys_subs) do
      spol := Groebner[SPolynomial](polys_subs[idx], polys_subs[jdx], tdeg(op(vars)), characteristic=2^29-3);
      reduct := Groebner[NormalForm](spol, polys_subs, tdeg(op(vars)), characteristic=2^29-3):
      if reduct = 0 then
        num_zero_reduct := num_zero_reduct+1:
      fi:
      if reduct <> 0 then
        num_monom := num_monom + nops(MonomialList(reduct)):
      fi:
    end do;
  od:
  finish := []:
  
  for attempt from 1 to 5 do
    polys_subs := map2(subs, {seq(vars[i]=vars[i]^LLL[i], i=1..numelems(vars))}, polysys);
    
    start := time[real]():
    gb := Groebner[Basis](polys_su bs, tdeg(op(vars)), characteristic=2^29-3):
    finish := [op(finish), time[real]() - start]:

    polys_subs := map2(subs, {seq(vars[i]^LLL[i]=vars[i], i=1..numelems(vars))}, polysys);
    gb := Groebner[Basis](polys_subs, tdeg(op(vars)), characteristic=2^29-3):
  end do;
  printf("%a, num_zero_reduct = %a, num_monom = %a, mean gb_time = %a, nops(gb) = %a\n", LLL, num_zero_reduct, num_monom, Statistics[Mean](finish), nops(gb)):
  print(finish):
od:
end proc:



