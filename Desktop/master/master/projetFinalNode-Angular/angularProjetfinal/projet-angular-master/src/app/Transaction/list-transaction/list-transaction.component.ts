import { Devise } from './../../class/devise';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';
import { Component, OnInit } from '@angular/core';
import { Transaction } from 'src/app/class/transaction';

@Component({
  selector: 'app-list-transaction',
  templateUrl: './list-transaction.component.html',
  styleUrls: ['./list-transaction.component.css']
})
export class ListTransactionComponent implements OnInit{
  constructor(private router:Router,private http:HttpClient){

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
    this.http
    .get<Transaction[]|any>(
      "http://localhost:3000/api/findAllTransaction",
      )
      .subscribe(res=>{
 this.donneTransaction=res.data
console.log(this.donneTransaction)
      })
  }

}
