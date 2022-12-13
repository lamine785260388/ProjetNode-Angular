import { Router } from '@angular/router';
import { OnInit } from '@angular/core';
import { Component } from '@angular/core';
import { NgForm } from '@angular/forms';
import { Service } from 'src/app/node.service';
import { HttpClient, HttpHeaders } from "@angular/common/http";
import { Pays } from 'src/app/class/pays';

@Component({
  selector: 'app-send',
  templateUrl: './send.component.html',
  styleUrls: ['./send.component.css']
})
export class SendComponent implements OnInit {
constructor(private router:Router,private http: HttpClient){
 if(sessionStorage.getItem('isloggin')!='true'){
  sessionStorage.setItem('url','send')
  this.router.navigate(['login'])
 }
 }
 httpOptions = {
  headers: new HttpHeaders({
    "Content-Type": "application/json"
  })
};
  ngOnInit(): void {

    console.log(sessionStorage.getItem('isloggin'))
    this.http
    .get(
      "http://localhost:3000/api/findAllPays",
      )
      .subscribe(res=>console.log(res))
   
   
  }
  
  submit (form: NgForm) {

    var user = form.value.username;
    var password=form.value.password;
    console.log('suis la');
     
   }
}
