import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { User } from 'src/app/class/user';

@Component({
  selector: 'app-liste-user',
  templateUrl: './liste-user.component.html',
  styleUrls: ['./liste-user.component.css']
})
export class ListeUserComponent implements OnInit {
  constructor(private http:HttpClient){}
  donne:User[]|any
  ngOnInit(): void {
    this.http
    .get<User[]|any>(
      "http://localhost:3000/api/findAllUser",
      )
      .subscribe(res=>{
this.donne=res.data
console.log(this.donne[0])
      })
    
  }

}
