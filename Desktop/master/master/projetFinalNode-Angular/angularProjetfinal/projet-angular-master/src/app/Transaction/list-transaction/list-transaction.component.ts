import { Devise } from './../../class/devise';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';
import { Component, OnInit } from '@angular/core';
import { Transaction } from 'src/app/class/transaction';
import { TransactionServiceService } from '../transaction-service.service';

@Component({
  selector: 'app-list-transaction',
  templateUrl: './list-transaction.component.html',
  styleUrls: ['./list-transaction.component.css']
})
export class ListTransactionComponent implements OnInit{
  constructor(private router:Router,private http:HttpClient,private transactionservice:TransactionServiceService){

    if(sessionStorage.getItem('isloggin')!='true'){
      sessionStorage.setItem('url','listTransaction')
      this.router.navigate(['login'])
     }
  }
  donneTransaction:Transaction[]
  httpOptions = {
    headers: new HttpHeaders({
      "Content-Type": "application/json"
    })
  };
  ngOnInit(): void {
    this.transactionservice.getTransaction()
      .subscribe(res=>{
 this.donneTransaction=res.data
console.log(this.donneTransaction)
      })
  }

}
