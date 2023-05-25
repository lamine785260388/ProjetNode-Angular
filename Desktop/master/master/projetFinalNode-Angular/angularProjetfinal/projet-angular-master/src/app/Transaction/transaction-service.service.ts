import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { Transaction } from '../class/transaction';
import { BaseUrl } from '../class/baseurl';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class TransactionServiceService {
  url=new BaseUrl;
  racine=this.url.url;

  constructor(private http:HttpClient) { }
  getTransaction():Observable<Transaction[]|any>{
    return this.http.get<Transaction[]>(this.racine+"findAllTransaction")
  }
}
